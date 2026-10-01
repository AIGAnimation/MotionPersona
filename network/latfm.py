"""Stage-2: flow matching in the stage-1 codec's token space.

Given 1 history token (same encoder over the past frames) + condition tokens (traj/text/betas), jointly
denoise the 3 future tokens with the shared FM core (network/fm.py); sampling = a few Euler/Heun steps in
z + one decode + stitch after the true prefix.  The frozen stage-1 VAE is rebuilt from its checkpoint's own
cfg and lives OUTSIDE `self.model`, so EMA / the optimizer / DDP-sanity / best-ckpt reload see exactly the
trainable LatentDenoiser.  Latents are normalised by per-dim statistics calibrated once at fit start from a
deterministic batch (buffers under `self.model` -> they ride along in every checkpoint).

Shape conditioning: `use_shape: true` = a betas token in the prior (the paper's prior); the shape-invariant codec
variant sets `use_shape: false` (betas live in the stage-1 decoder only).  continuity is a raw-space concept and
is rejected here -- seam continuity comes from the decoder's past_tail conditioning.  Export semantics: 'pred' =
decoded one-pass z1_hat at a random t; 'sample' = decoded ODE rollout.  History tokens are encoded from
batch['data'][..., :K] (real frames incl. the contact row) in both training and sampling.

Denoiser variants (all default off):
`diffusion.arch: dit` = LatentDiT (pre-LN adaLN-Zero blocks, time via modulation instead of a token), with
`pos: rope` = rotary attention on frame-time positions shared by traj / history / z tokens; `t_sampling:
logit_normal` = SD3-style mid-t emphasis when drawing the training t.

Split conditioning: `model.style_cond: embed` adds a learned per-style token (StyleEmbed) and
`data.text_source: persona_feat` makes the text token the CLIP feature of a persona-only prompt, so persona and
style are two tokens instead of one fused prompt.  Token order [time, shape, style, text, traj_trans, traj_pose,
hist, z...] (DiT: no time token).  Only the history token is ever masked (cond_mask_prob).

In-betweening switch (default OFF): `data.end_token_prob` > 0 adds ONE
END token, built exactly like the history token -- the frozen codec encoder over the window's last `token_stride`
frames, z-normalised, through its own nn.Linear -- inserted between the history token and the z tokens.  The value is
P(the token is visible) per training sample (1.0 = always, the in-betweening application); a masked sample
zeroes the pre-embedding vector, so the Linear's bias is the learned "no end keyframe" token, the same mechanism
`cond_mask_prob` uses for the history.  With the default 0.0 no module is constructed, no token is inserted and no RNG
is consumed, so checkpoints trained without it load strict.  Inference-only
alternative that needs no retraining: `sample(..., z_pin=(idx, value))` pins z tokens during the ODE (token-space
inpainting).

Persona tokens (paper section 5): `model.persona_cond` replaces the CLIP text token by learned tokens --
`id` = one nn.Embedding row per performer (SUBJECT_VOCAB, 44), `attr` = three typed tables (role / affiliation /
dominance), `id_attr` = all four tokens (the paper's prior), `none` = no persona token (ablation); `text` (default)
keeps the CLIP token.  Token order [time, shape, style, persona..., traj_trans,
traj_pose, hist, z...].  Persona tokens are never masked either.

Audio stream (speech-driven gesture application, default OFF): `model.cond_stream:
audio` replaces the two trajectory token streams by ONE per-frame audio token stream (conditions['audio_feat'] (B, D, Fa),
pre-extracted speech features over the clip window + `data.audio_lookahead` frames) through a 2-layer MLP; no trajectory
module is constructed then.  `traj` (the default / an absent key) builds the trajectory modules, so every
locomotion checkpoint loads strict.  The learned ID / style tables take
their sizes from a dataset-provided vocabulary when meta carries one (network/cond_tokens.py vocab_sizes).
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint

from network.checkpoint import load_weights
from network.cond_tokens import (PERSONA_CONDS, PERSONA_LEARNED, STYLE_CONDS, PersonaEmbed, StyleEmbed, persona_tokens,
                                 vocab_sizes)
from network.fm import fm_bin_to_t, fm_interpolate, fm_sample_loop, fm_x1_from_v
from network.lit_module import MotionDiffusionModule
from network.models import PositionalEncoding, TimestepEmbedder, TrajProcess
from network.vae import build_vae



COND_STREAMS = ('traj', 'audio')


class AudioProcess(nn.Module):
    """`model.cond_stream: audio`: per-frame speech features (B, D, F) -> F tokens [F, B, d] (the TrajProcess layout)
    through Linear-GELU-Linear (the features are 770-d HuBERT + prosody, a single Linear would be a weak adapter)."""

    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_feats, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))

    def forward(self, x):
        return self.net(x.permute(2, 0, 1))


def _stream_tokens(model, traj_pose, traj_trans, audio):
    """The per-frame condition stream: [traj_trans, traj_pose] tokens (default) or the audio tokens."""
    if getattr(model, 'cond_stream', 'traj') == 'audio':
        assert audio is not None, 'model.cond_stream=audio needs conditions[\'audio_feat\'] (data.audio=true)'
        return [model.audio_embed(audio)]
    return [model.traj_trans_embed(traj_trans), model.traj_pose_embed(traj_pose)]


class LatentDenoiser(nn.Module):
    """Token-space v-predictor: [time, (shape), (style), persona..., traj_trans, traj_pose, hist, (end), z_t] -> v for the z tokens."""

    def __init__(self, d_z, n_tokens, shape_dim, latent_dim, ff_size, num_layers, num_heads, dropout, use_shape=True,
                 style_cond='none', persona_cond='text', end_token=False, cond_stream='traj', audio_dim=0, n_subjects=None,
                 n_styles=None):
        super().__init__()
        self.d_z, self.n_tokens, self.use_shape = d_z, n_tokens, use_shape
        self.cond_stream = str(cond_stream)
        self.use_style = style_cond == 'embed'
        self.persona_cond, self.use_text = persona_cond, persona_cond == 'text'
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.z_embed = nn.Linear(d_z, latent_dim)
        self.hist_embed = nn.Linear(d_z, latent_dim)
        self.end_token = bool(end_token)
        if self.end_token:
            self.end_embed = nn.Linear(d_z, latent_dim)             # in-betweening switch: same shape as hist_embed
        if self.use_text:
            self.text_embed = TrajProcess(512, latent_dim)
        self.persona = PersonaEmbed(latent_dim, persona_cond, n_subjects=n_subjects) if persona_cond in PERSONA_LEARNED else None
        if self.cond_stream == 'audio':
            self.audio_embed = AudioProcess(int(audio_dim), latent_dim)   # no traj module at all (DDP: no unused params)
        else:
            self.traj_pose_embed = TrajProcess(6, latent_dim)
            self.traj_trans_embed = TrajProcess(2, latent_dim)
        if use_shape:
            self.shape_embed = TrajProcess(shape_dim, latent_dim)   # only constructed when used (DDP)
        if self.use_style:
            self.style_embed = StyleEmbed(latent_dim, n_styles=n_styles)
        layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                                           dropout=dropout, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(latent_dim, d_z)
        self.register_buffer('z_mean', torch.zeros(d_z))
        self.register_buffer('z_std', torch.ones(d_z))
        self.register_buffer('z_calibrated', torch.zeros(1))

    def forward(self, z_t, timesteps, hist, traj_pose, traj_trans, shape_feat, text_feat, style_idx=None, persona=None,
                end=None, audio=None):
        n = z_t.shape[1]
        toks = [self.embed_timestep(timesteps)]
        if self.use_shape:
            assert shape_feat is not None, 'use_shape=true needs shape_feat'
            toks.append(self.shape_embed(shape_feat))
        if self.use_style:
            toks.append(self.style_embed(style_idx))
        toks += persona_tokens(self, text_feat, persona)
        stream = _stream_tokens(self, traj_pose, traj_trans, audio)
        toks += stream + [self.hist_embed(hist).permute(1, 0, 2)]
        if self.end_token:
            assert end is not None, 'data.end_token_prob > 0 needs the end token'
            toks.append(self.end_embed(end).permute(1, 0, 2))
        toks.append(self.z_embed(z_t).permute(1, 0, 2))
        h = self.enc(self.sequence_pos_encoder(torch.cat(toks, dim=0)))[-n:]
        return self.out(h).permute(1, 0, 2)                          # [B, n, d_z]


# ---------------------------------------------------------------------------- arch: dit
def rope_rotate(x, pos, base=10000.0):
    """Rotary embedding on real-valued positions. x [B, H, N, dh], pos [N] float (frames; 0 = no rotation).
    Standard rotate-half pairing over dh/2 frequencies base^(-2j/dh); relative-position property holds for any
    real offsets, so tokens from different streams can share one time axis."""
    dh = x.shape[-1]
    inv = base ** (-torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)      # [dh/2]
    ang = pos.to(torch.float32)[:, None] * inv[None]                                       # [N, dh/2]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RotaryAttention(nn.Module):
    """Multi-head self-attention, batch-first, with optional rotary positions (pos=None: plain attention)."""

    def __init__(self, d, heads, dropout, rope_base):
        super().__init__()
        assert d % heads == 0 and (d // heads) % 2 == 0, (d, heads)
        self.heads, self.dropout, self.rope_base = heads, dropout, rope_base
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, pos=None):
        B, N, d = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)   # [3, B, H, N, dh]
        if pos is not None:
            q, k = rope_rotate(q, pos, self.rope_base), rope_rotate(k, pos, self.rope_base)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).reshape(B, N, d))


def _modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """Pre-LN transformer block with adaLN-Zero time conditioning (Peebles & Xie 2023): the 6 modulation
    vectors come from the time embedding, zero-initialised so every block starts as the identity."""

    def __init__(self, d, heads, ff, dropout, rope_base):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = RotaryAttention(d, heads, dropout, rope_base)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(approximate='tanh'), nn.Dropout(dropout), nn.Linear(ff, d),
                                 nn.Dropout(dropout))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, pos=None):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        x = x + g1[:, None] * self.attn(_modulate(self.norm1(x), sh1, sc1), pos)
        return x + g2[:, None] * self.mlp(_modulate(self.norm2(x), sh2, sc2))


class LatentDiT(nn.Module):
    """`diffusion.arch: dit` -- same inputs/outputs as LatentDenoiser, but: time enters through adaLN-Zero instead
    of a token, blocks are pre-LN, and `pos: rope` replaces the additive sinusoidal token-index encoding with
    rotary attention on FRAME-TIME positions shared by every stream (traj frame f -> f, hist token -> centre of
    its frames, z token i -> centre of frames K + s*i .. K + s*(i+1); shape/text at 0 = unrotated), so relative
    position = time offset in frames across modalities.  `pos: abs` keeps the token-index sinusoid."""

    def __init__(self, d_z, n_tokens, shape_dim, latent_dim, ff_size, num_layers, num_heads, dropout, use_shape=True,
                 pos='abs', rope_base=10000.0, past_frame=10, stride=5, grad_ckpt=False, style_cond='none',
                 persona_cond='text', end_token=False, cond_stream='traj', audio_dim=0, n_subjects=None, n_styles=None):
        super().__init__()
        assert pos in ('abs', 'rope'), pos
        self.d_z, self.n_tokens, self.use_shape, self.pos = d_z, n_tokens, use_shape, pos
        self.cond_stream = str(cond_stream)
        self.use_style = style_cond == 'embed'
        self.persona_cond, self.use_text = persona_cond, persona_cond == 'text'
        self.grad_ckpt = bool(grad_ckpt)     # recompute each block in backward: 8x384 @1024 = 23 GB -> fits a shared card
        self.past_frame, self.stride = int(past_frame), int(stride)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)      # sinusoid table (time embed + abs)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)
        self.z_embed = nn.Linear(d_z, latent_dim)
        self.hist_embed = nn.Linear(d_z, latent_dim)
        self.end_token = bool(end_token)
        if self.end_token:
            self.end_embed = nn.Linear(d_z, latent_dim)             # in-betweening switch: same shape as hist_embed
        if self.use_text:
            self.text_embed = TrajProcess(512, latent_dim)
        self.persona = PersonaEmbed(latent_dim, persona_cond, n_subjects=n_subjects) if persona_cond in PERSONA_LEARNED else None
        if self.cond_stream == 'audio':
            self.audio_embed = AudioProcess(int(audio_dim), latent_dim)   # no traj module at all (DDP: no unused params)
        else:
            self.traj_pose_embed = TrajProcess(6, latent_dim)
            self.traj_trans_embed = TrajProcess(2, latent_dim)
        if use_shape:
            self.shape_embed = TrajProcess(shape_dim, latent_dim)
        if self.use_style:
            self.style_embed = StyleEmbed(latent_dim, n_styles=n_styles)
        self.blocks = nn.ModuleList([DiTBlock(latent_dim, num_heads, ff_size, dropout, rope_base) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(latent_dim, 2 * latent_dim))
        self.out = nn.Linear(latent_dim, d_z)
        for lin in (self.ada_out[1], self.out):                                  # DiT final-layer zero init
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.register_buffer('z_mean', torch.zeros(d_z))
        self.register_buffer('z_std', torch.ones(d_z))
        self.register_buffer('z_calibrated', torch.zeros(1))

    def positions(self, n_global, n_traj, n_hist, n_z, device, n_end=0, n_audio=0):
        """Frame-time position of every token in sequence order (see class doc).  n_end (the in-betweening switch)
        adds the end token between hist and z, at the centre of the window's last `stride` frames."""
        K, s = self.past_frame, self.stride
        window = K + s * n_z
        traj = torch.arange(n_traj, device=device, dtype=torch.float32) * (window / n_traj)
        hist = K - (n_hist - torch.arange(n_hist, device=device, dtype=torch.float32)) * s + (s - 1) / 2
        z = K + torch.arange(n_z, device=device, dtype=torch.float32) * s + (s - 1) / 2
        end = torch.full((n_end,), window - s + (s - 1) / 2, device=device, dtype=torch.float32)
        if n_audio:                                   # audio stream: frame f of the window (+ lookahead) sits at f
            aud = torch.arange(n_audio, device=device, dtype=torch.float32)
            return torch.cat([torch.zeros(n_global, device=device), aud, hist, end, z])
        return torch.cat([torch.zeros(n_global, device=device), traj, traj, hist, end, z])

    def forward(self, z_t, timesteps, hist, traj_pose, traj_trans, shape_feat, text_feat, style_idx=None, persona=None,
                end=None, audio=None):
        n = z_t.shape[1]
        c = self.embed_timestep(timesteps)[0]                                   # [1, B, d] -> [B, d]
        toks = []
        if self.use_shape:
            assert shape_feat is not None, 'use_shape=true needs shape_feat'
            toks.append(self.shape_embed(shape_feat))
        if self.use_style:
            toks.append(self.style_embed(style_idx))
        toks += persona_tokens(self, text_feat, persona)
        n_global = len(toks)                                                   # shape / style / persona tokens, all at frame 0
        stream = _stream_tokens(self, traj_pose, traj_trans, audio)
        toks += stream + [self.hist_embed(hist).permute(1, 0, 2)]
        n_end = 0
        if self.end_token:
            assert end is not None, 'data.end_token_prob > 0 needs the end token'
            toks.append(self.end_embed(end).permute(1, 0, 2))
            n_end = end.shape[1]
        toks.append(self.z_embed(z_t).permute(1, 0, 2))
        x = torch.cat(toks, dim=0)                                             # [N, B, d]
        pos = None
        if self.pos == 'rope':
            if self.cond_stream == 'audio':
                pos = self.positions(n_global, 0, hist.shape[1], n, x.device, n_end=n_end, n_audio=audio.shape[-1])
            else:
                pos = self.positions(n_global, traj_trans.shape[-1], hist.shape[1], n, x.device, n_end=n_end)
            x = self.sequence_pos_encoder.dropout(x)
        else:
            x = self.sequence_pos_encoder(x)
        x = x.permute(1, 0, 2)                                                 # [B, N, d]
        ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        for blk in self.blocks:
            x = checkpoint(blk, x, c, pos, use_reentrant=False) if ckpt else blk(x, c, pos)
        sh, sc = self.ada_out(c).chunk(2, dim=-1)
        return self.out(_modulate(self.norm_out(x[:, -n:]), sh, sc))            # [B, n, d_z]


def resolve_vae_ckpt(path):
    """`diffusion.vae_ckpt` as recorded in a stage-2 checkpoint, or its new location after the run directory
    was moved into a sub-folder of the same save root (save/<run> -> save/<group>/.../<run>): run names are
    unique, so `<root>/**/<run>/<file>` with exactly one hit is unambiguous.  Raises SystemExit otherwise."""
    if os.path.exists(path):
        return path
    # Anchor the search at the save root (`<save>/<run>/<file>` -> parents[1]), not at parts[0]: for an ABSOLUTE
    # recorded path that would be '/', i.e. a scan of the whole filesystem -- which on macOS also finds every hit
    # twice through the /System/Volumes/Data firmlink and then fails as "2 candidates".
    p = Path(path)
    root = p.parents[1] if len(p.parts) >= 3 else Path(p.parts[0])
    hits = sorted(root.glob(f'**/{p.parent.name}/{p.name}')) if len(p.parts) >= 3 and root.is_dir() else []
    if len(hits) == 1:
        return str(hits[0])
    raise SystemExit(f'diffusion.vae_ckpt not found: {path} (train the stage-1 vae first)'
                     + (f'; {len(hits)} candidates under {p.parts[0]}: {hits}' if hits else ''))


class LatentFMModule(MotionDiffusionModule):
    def __init__(self, cfg, meta):
        super().__init__(cfg, meta)
        d = cfg.diffusion
        assert str(d.type) == 'latfm', d.type
        if self.delta:
            raise SystemExit('diffusion.delta is not supported by the latfm head')
        if self.continuity.enabled:
            raise SystemExit('continuity is raw-space; the latfm seam story is the stage-1 decoder past_tail')
        vae_ckpt = resolve_vae_ckpt(str(d.vae_ckpt))
        ck = torch.load(vae_ckpt, map_location='cpu', weights_only=False)
        vcfg = OmegaConf.create(ck['hyper_parameters']['cfg'])
        self.vae = build_vae(vcfg, meta)
        load_weights(self.vae, vae_ckpt)
        self.vae.requires_grad_(False)
        self.vae.eval()
        vd = vcfg.diffusion
        self.n_tokens, self.d_z, self.vae_n_tail = int(vd.n_future_tokens), int(vd.d_z), int(vd.n_tail)
        self.vae_dec_shape = str(vd.decoder_cond).endswith('+shape')
        self.vae_enc_shape = str(vd.get('encoder_cond', 'none')) == 'shape'
        self.use_shape = bool(d.get('use_shape', True))
        self.style_cond = str(cfg.model.get('style_cond', 'none'))    # absent key -> none
        if self.style_cond not in STYLE_CONDS:
            raise SystemExit(f'model.style_cond must be none | embed, got {self.style_cond}')
        self.use_style = self.style_cond == 'embed'
        self.persona_cond = str(cfg.model.get('persona_cond', 'text'))  # absent key -> text
        if self.persona_cond not in PERSONA_CONDS:
            raise SystemExit(f'model.persona_cond must be one of {PERSONA_CONDS}, got {self.persona_cond}')
        self.use_text = self.persona_cond == 'text'
        # speech-driven gesture switch: absent key / 'traj' = the locomotion prior
        self.cond_stream = str(cfg.model.get('cond_stream', 'traj'))
        if self.cond_stream not in COND_STREAMS:
            raise SystemExit(f'model.cond_stream must be one of {COND_STREAMS}, got {self.cond_stream}')
        stream_kw = {}
        if self.cond_stream == 'audio':
            if not meta.get('audio_dim'):
                raise SystemExit('model.cond_stream=audio needs a dataset with frames/audio_feat.npy (meta audio_dim)')
            if not bool(cfg.data.get('audio', False)):
                raise SystemExit('model.cond_stream=audio needs data.audio=true (the loader must serve audio_feat)')
            stream_kw = dict(cond_stream='audio', audio_dim=int(meta['audio_dim']))
        n_subj, n_sty = vocab_sizes(meta)
        if n_subj is not None:
            stream_kw['n_subjects'] = n_subj
        if n_sty is not None:
            stream_kw['n_styles'] = n_sty
        self.time_bins, self.fm_steps, self.fm_sampler = int(d.time_bins), int(d.sample_steps), str(d.sampler)
        self.z_posterior = bool(d.z_posterior_sample)
        # ---- in-betweening switch; 0.0 = off (default)
        self.end_token_prob = float(cfg.data.get('end_token_prob', 0.0))
        if not 0.0 <= self.end_token_prob <= 1.0:
            raise SystemExit(f'data.end_token_prob must be in [0, 1], got {self.end_token_prob}')
        self.end_token = self.end_token_prob > 0.0
        # frames featurize_batch must hand back as conditions['end_motion'] -- exactly one codec token's worth
        self.end_frames = int(vd.token_stride) if self.end_token else 0
        dims = (self.d_z, self.n_tokens, int(meta['shape_dim']), int(cfg.model.latent_dim), int(cfg.model.ff_size),
                int(cfg.model.num_layers), int(cfg.model.num_heads), float(cfg.model.dropout))
        arch = str(d.get('arch', 'token'))                            # absent key -> token
        if arch == 'token':
            self.model = LatentDenoiser(*dims, use_shape=self.use_shape, style_cond=self.style_cond,
                                        persona_cond=self.persona_cond, end_token=self.end_token, **stream_kw)
        elif arch == 'dit':
            self.model = LatentDiT(*dims, use_shape=self.use_shape, pos=str(d.get('pos', 'abs')),
                                   rope_base=float(d.get('rope_base', 10000.0)), past_frame=self.past_frame,
                                   stride=int(vd.token_stride), grad_ckpt=bool(d.get('grad_ckpt', False)),
                                   style_cond=self.style_cond, persona_cond=self.persona_cond,
                                   end_token=self.end_token, **stream_kw)
        else:
            raise SystemExit(f'diffusion.arch must be token | dit, got {arch}')
        self.t_sampling = str(d.get('t_sampling', 'uniform'))
        if self.t_sampling not in ('uniform', 'logit_normal'):
            raise SystemExit(f'diffusion.t_sampling must be uniform | logit_normal, got {self.t_sampling}')
        self.t_ln = (float(d.get('t_ln_mean', 0.0)), float(d.get('t_ln_std', 1.0)))
        self.cond_mask_prob = float(cfg.model.cond_mask_prob)        # history-token CFG dropout

    # ---------------------------------------------------------------- plumbing
    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()                                              # Lightning flips modes every epoch
        return self

    def _norm(self, z):
        return (z - self.model.z_mean) / self.model.z_std

    def _denorm(self, z):
        return z * self.model.z_std + self.model.z_mean

    def _vae_shape(self, cond, dec):
        return cond['shape_feat'] if (self.vae_dec_shape if dec else self.vae_enc_shape) else None

    def _style(self, cond):
        """The style token's index (never masked); None when the prior has no style token."""
        if not self.use_style:
            return None
        if 'style_idx' not in cond:
            raise SystemExit('model.style_cond=embed needs style_idx in the batch (LocoDataset clips carry it from meta label.style)')
        return cond['style_idx']

    def _persona(self, cond):
        """(subject_idx, attr_idx) for the learned persona tokens (never masked); None for persona_cond text / none."""
        if self.persona_cond not in PERSONA_LEARNED:
            return None
        need = (['subject_idx'] if self.persona_cond != 'attr' else []) + (['attr_idx'] if self.persona_cond != 'id' else [])
        missing = [k for k in need if k not in cond]
        if missing:
            raise SystemExit(f'model.persona_cond={self.persona_cond} needs {missing} in the batch '
                             '(LocoDataset clips carry them from manifest subject / meta label)')
        return cond.get('subject_idx'), cond.get('attr_idx')

    def _stream_kw(self, cond):
        """{'audio': (B, D, Fa)} for the audio stream, {} for the default trajectory stream (unchanged call)."""
        if self.cond_stream != 'audio':
            return {}
        if 'audio_feat' not in cond:
            raise SystemExit('model.cond_stream=audio needs conditions[\'audio_feat\'] in the batch (data.audio=true)')
        return {'audio': cond['audio_feat']}

    def _text(self, cond):
        return cond.get('text_feat') if self.use_text else None

    def _end(self, cond):
        """The END token (in-betweening switch): the frozen codec encoder over the window's last `token_stride`
        real frames -- literally `encode_past` on conditions['end_motion'], the same call the history token uses --
        then the same z normalisation.  None when the switch is off.  The key is only ever produced by
        featurize_batch(end_frames=self.end_frames), so a batch built for free rollout (whose 'future' rows are a
        repeat of the last history frame) fails loudly instead of conditioning on padding."""
        if not self.end_token:
            return None
        if 'end_motion' not in cond:
            raise SystemExit('data.end_token_prob > 0 needs conditions[\'end_motion\']: featurize the batch through '
                             'this module (module.featurize / on_after_batch_transfer), which passes end_frames='
                             f'{self.end_frames}. A free (no-future) rollout batch cannot feed this model.')
        end = cond['end_motion']
        if end.shape[-1] != self.end_frames:
            raise SystemExit(f'conditions[\'end_motion\'] has {end.shape[-1]} frames, expected {self.end_frames}')
        with torch.no_grad():
            mu = self.vae.encode_past(end, shape_feat=self._vae_shape(cond, dec=False))
        return self._norm(mu)

    def on_fit_start(self):
        if float(self.model.z_calibrated) > 0:
            return
        # Private RNG + explicit AugParams (mirrored half the time, like training): zero side effects on the
        # dataset's shared aug RNG, and a different batch from TrainMonitor's export clips.
        import numpy as np
        from torch.utils.data import default_collate
        from data.loco_dataset import AugParams
        ds = self.trainer.datamodule.train_set
        rng = np.random.RandomState(int(self.cfg.export.seed) + 1)
        n = min(256, len(ds))
        raws = [ds.get_raw_item(int(rng.randint(len(ds))),
                                AugParams(float(rng.uniform(0, 2 * np.pi)), int(rng.randint(2)),
                                          float(rng.rand()), int(rng.randint(3)), mirror=bool(rng.randint(2))))
                for _ in range(n)]
        batch = self.featurize(default_collate(raws))
        x0 = batch['data'].permute(0, 2, 3, 1)
        with torch.no_grad():
            mu, _ = self.vae.encode(x0[..., self.past_frame:], shape_feat=self._vae_shape(batch['conditions'], dec=False))
        z = mu.reshape(-1, self.d_z)
        self.model.z_mean.copy_(z.mean(0))
        self.model.z_std.copy_(z.std(0).clamp(min=1e-4))
        self.model.z_calibrated.fill_(1.0)
        print(f'latfm: z calibrated on {z.shape[0]} tokens | std range {float(self.model.z_std.min()):.3f}..{float(self.model.z_std.max()):.3f}')

    def _encode_targets(self, batch, need_z1=True):
        """`need_z1=False` skips the encoder pass over the future window: sampling only needs the history token, and
        at inference the 'future' rows are padding anyway (12 % of a realtime chunk, bit-exact -- the posterior draw
        that would consume RNG only happens in training)."""
        cond = batch['conditions']
        x0 = batch['data'].permute(0, 2, 3, 1)
        K = self.past_frame
        with torch.no_grad():
            z1 = None
            if need_z1:
                mu, logvar = self.vae.encode(x0[..., K:], shape_feat=self._vae_shape(cond, dec=False))
                if self.z_posterior and self.training:
                    z1 = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                else:
                    z1 = mu
            hist = self.vae.encode_past(x0[..., :K], shape_feat=self._vae_shape(cond, dec=False))
        return x0, (None if z1 is None else self._norm(z1)), self._norm(hist)

    # ---------------------------------------------------------------- contract
    def forward_diffuse(self, batch, **kw):
        cond = batch['conditions']
        x0, z1, hist = self._encode_targets(batch)
        batch_size = z1.shape[0]
        noise = kw.get('noise')
        if noise is None:
            noise = torch.randn_like(z1)
        t_idx = kw.get('t_idx')
        if t_idx is None:
            if self.t_sampling == 'logit_normal':                     # SD3: t = sigmoid(N(m, s)) -> mid-t emphasis
                u = torch.sigmoid(torch.randn(batch_size, device=z1.device) * self.t_ln[1] + self.t_ln[0])
                t_idx = (u * self.time_bins).long().clamp_(0, self.time_bins - 1)
            else:
                t_idx = torch.randint(0, self.time_bins, (batch_size,), device=z1.device)
        t_cont = fm_bin_to_t(t_idx, self.time_bins)
        z_t = fm_interpolate(z1, noise, t_cont)
        if self.training and self.cond_mask_prob > 0:
            keep = torch.rand(batch_size, device=z1.device) < (1 - self.cond_mask_prob)
        else:
            keep = torch.ones(batch_size, dtype=torch.bool, device=z1.device)
        hist_in = hist * keep.view(batch_size, 1, 1)                 # the only masked condition (style/text/traj never)
        end_in = self._end(cond)
        if end_in is not None and self.training and self.end_token_prob < 1.0:
            # P(visible) per sample; a zeroed pre-embedding vector makes end_embed's bias the learned null token
            # (the cond_mask_prob mechanism).  No draw at all at prob 1.0, and none when the switch is off.
            vis = (torch.rand(batch_size, device=z1.device) < self.end_token_prob).to(end_in.dtype)
            end_in = end_in * vis.view(batch_size, 1, 1)
        v_pred = self.model(z_t, t_idx, hist_in, cond['traj_pose'], cond['traj_trans'],
                            cond['shape_feat'] if self.use_shape else None, self._text(cond), style_idx=self._style(cond),
                            persona=self._persona(cond), **({'end': end_in} if end_in is not None else {}),
                            **self._stream_kw(cond))
        z1_hat = fm_x1_from_v(z_t, v_pred, t_cont)
        return {
            'model_output': z1_hat, 'x0': x0, 'x_t': z_t, 't': t_idx, 'noise': noise, 'keep': keep,
            'is_clean': None, 'gt_full': x0, 'pred_full': None,
            'v_pred': v_pred, 'v_target': z1 - noise, 'z1': z1, 'z1_hat': z1_hat, 't_cont': t_cont,
        }

    def _losses(self, out, cond):
        loss_v = torch.mean((out['v_pred'] - out['v_target']) ** 2)
        return {'loss_v': loss_v, 'loss': loss_v}

    def _decode_full(self, z_norm, x0, cond):
        K = self.past_frame
        tail = x0[..., K - self.vae_n_tail:K] if 'past_tail' in self.vae.decoder_cond else None
        fut = self.vae.decode(self._denorm(z_norm), past_tail=tail, shape_feat=self._vae_shape(cond, dec=True))
        return torch.cat([x0[..., :K], fut], dim=-1)

    @torch.no_grad()
    def x0_at_random_t(self, batch):
        out = self.forward_diffuse(batch)
        pred = self._decode_full(out['z1_hat'], out['x0'], batch['conditions'])
        return out['gt_full'].permute(0, 3, 1, 2), pred.permute(0, 3, 1, 2)

    @torch.no_grad()
    def sample(self, batch, generator=None, *, x_init=None, shape_prior=None, shape_dec=None, z_pin=None):
        """Few-step ODE rollout in z + one decode. `x_init` (B, n_tokens, d_z) replaces the generator's initial noise
        (the evaluation engine draws it per case). `shape_prior` / `shape_dec` (B, S, 1) replace `cond['shape_feat']`
        for the prior and for the decoder respectively (which stage carries a body-shape effect); None keeps the
        batch's own shape, so the default call is unchanged.
        `z_pin=(idx, value)` (LongTensor of token indices, NORMALISED values (B, len(idx), d_z)) holds those z tokens
        fixed after every ODE update = token-space inpainting, which turns ANY existing prior into an in-betweening
        model with no retraining; None = the default, unchanged path."""
        cond = batch['conditions']
        x0, _, hist = self._encode_targets(batch, need_z1=False)
        batch_size = x0.shape[0]

        style, persona, text = self._style(cond), self._persona(cond), self._text(cond)
        shape_p = cond['shape_feat'] if shape_prior is None else shape_prior
        end = self._end(cond)
        end_kw = {'end': end} if end is not None else {}
        end_kw.update(self._stream_kw(cond))

        def v_fn(z, t_idx):
            return self.guided_by_history(
                lambda h: self.model(z, t_idx, h, cond['traj_pose'], cond['traj_trans'],
                                     shape_p if self.use_shape else None, text, style_idx=style, persona=persona,
                                     **end_kw), hist)

        pin_fn = None
        if z_pin is not None:
            pin_idx, pin_val = z_pin

            def pin_fn(z, _i=pin_idx, _v=pin_val):
                z = z.clone()
                z[:, _i] = _v.to(z.dtype)
                return z

        z = fm_sample_loop(v_fn, (batch_size, self.n_tokens, self.d_z), time_bins=self.time_bins,
                           steps=self.fm_steps, sampler=self.fm_sampler, generator=generator, device=x0.device,
                           pin_fn=pin_fn, x_init=x_init)
        cond_dec = cond if shape_dec is None else dict(cond, shape_feat=shape_dec)
        out = self._decode_full(z, x0, cond_dec)
        return x0.permute(0, 3, 1, 2), out.permute(0, 3, 1, 2)
