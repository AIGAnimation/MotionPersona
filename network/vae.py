"""Stage-1 KL-VAE motion codec. Temporal tokens: the 45-frame future window compresses to n_future_tokens
tokens of d_z at token_stride frames each (configs/diffusion/vae.yaml: 3 x 15; the paper's codec: 9 x 5); the
10 past frames map to one history token via the same encoder (left edge-pad to one stride).

Shape handling: the encoder takes NO betas input (encoder_cond: none); decoder_cond: past_tail+shape (the paper's
codec) gives the decoder the body shape so it renders the motion on that body.  The reconstruction loss uses
skel_offset through the compute_losses stack either way (FK terms weigh errors in world space on the correct
body).  encoder_cond: shape + decoder_cond: past_tail+shape is the shape-invariant variant (VAEInvModule).

Export semantics: 'pred' and 'sample' are both the DETERMINISTIC reconstruction (z = mu), i.e. codec fidelity.
z will still carry shape-correlated content (root height etc.); acceptable for a compressor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F_nn

from network.checkpoint import PER_ROT_FEAT
from network.lit_module import MotionDiffusionModule
from network.models import MotionProcess, PositionalEncoding, TrajProcess


class TokenEncoder(nn.Module):
    """[B, J, F, n*stride] (+ optional shape token) -> (mu, logvar) [B, n, d_z]."""

    def __init__(self, input_feats, d_z, stride, latent_dim, num_layers, num_heads, ff_size, dropout, shape_dim=None):
        super().__init__()
        self.stride, self.d_z = stride, d_z
        self.patch = nn.Linear(stride * input_feats, latent_dim)
        self.shape_embed = TrajProcess(shape_dim, latent_dim) if shape_dim else None
        self.pos = PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                                           dropout=dropout, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(latent_dim, 2 * d_z)

    def forward(self, x, shape_feat=None):
        bs, njoints, nfeats, nframes = x.shape
        n = nframes // self.stride
        assert n * self.stride == nframes, (nframes, self.stride)
        p = x.permute(3, 0, 1, 2).reshape(nframes, bs, njoints * nfeats)
        p = p.reshape(n, self.stride, bs, njoints * nfeats).permute(0, 2, 1, 3).reshape(n, bs, self.stride * njoints * nfeats)
        seq = self.patch(p)                                          # [n, B, L]
        n_cond = 0
        if self.shape_embed is not None:
            assert shape_feat is not None, 'encoder_cond: shape needs shape_feat'
            seq = torch.cat([self.shape_embed(shape_feat), seq], dim=0)
            n_cond = 1
        h = self.enc(self.pos(seq))[n_cond:]                         # [n, B, L]
        mu, logvar = self.head(h).chunk(2, dim=-1)
        return mu.permute(1, 0, 2), logvar.permute(1, 0, 2)          # [B, n, d_z]


class TemporalRefine(nn.Module):
    """Residual depthwise-temporal refinement of the decoder's per-frame output.

    `TokenDecoder` in patch mode projects each token's hidden state to `stride` frames with ONE shared Linear, so
    nothing ties the last frame of patch i to the first frame of patch i+1: the measured jerk of the decoded feet
    spikes at every token boundary (period 5 = token_stride). These convolutions run
    ACROSS the frame axis with a receptive field wider than one patch, so a patch boundary is no longer a place
    where the model has no information. Depthwise over the feature channels keeps it tiny (~70k params at 150
    channels x 3 layers) and the residual starts it as a no-op.
    """

    def __init__(self, channels, layers=3, kernel=5):
        super().__init__()
        assert kernel % 2 == 1, kernel
        self.kernel = int(kernel)
        self.blocks = nn.ModuleList()
        for _ in range(int(layers)):
            self.blocks.append(nn.ModuleDict(dict(
                dw=nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels),
                norm=nn.LayerNorm(channels),                    # per FRAME, over channels: keeps the block local in
                pw1=nn.Conv1d(channels, channels, 1),           # time (a GroupNorm over [B,C,F] would normalise
                pw2=nn.Conv1d(channels, channels, 1))))         # across frames and couple the whole window)
        for b in self.blocks:                                   # zero-init the last projection: exact no-op at step 0
            nn.init.zeros_(b['pw2'].weight); nn.init.zeros_(b['pw2'].bias)

    def forward(self, x):
        """x: [B, C, F] -> [B, C, F]. Receptive field (kernel - 1) * layers // 2 frames each way."""
        for b in self.blocks:
            h = b['dw'](x)
            h = b['norm'](h.transpose(1, 2)).transpose(1, 2)
            x = x + b['pw2'](F_nn.gelu(b['pw1'](h)))
        return x


class TokenDecoder(nn.Module):
    """[B, n, d_z] (+ cond tokens) -> [B, J, F, n*stride]. cond: none | past_tail | past_tail+shape.

    `mode`:
      patch  one shared Linear turns each token's hidden state into `stride` frames. Fast, but the patch
             boundary every `stride` frames is where the period-5 jerk ripple comes from.
      query  `stride` learned frame queries per token are appended to the sequence and decoded per frame, so
             there is no patch projection and no patch boundary by construction. Sequence length grows from
             n (+cond) to n + n*stride (+cond), i.e. 11 -> 56 at n=9/stride=5 -- still trivial for attention.
    `refine_layers` > 0 adds the residual temporal head (TemporalRefine) on top of whichever mode is used.
    """

    def __init__(self, input_feats, njoints, nfeats, d_z, stride, latent_dim, num_layers, num_heads, ff_size,
                 dropout, cond='past_tail', n_tail=5, shape_dim=None, mode='patch', refine_layers=0,
                 refine_kernel=5):
        super().__init__()
        self.stride, self.njoints, self.nfeats, self.cond, self.n_tail = stride, njoints, nfeats, cond, n_tail
        self.mode = str(mode)
        assert self.mode in ('patch', 'query'), self.mode
        self.z_embed = nn.Linear(d_z, latent_dim)
        self.tail_embed = MotionProcess(input_feats, latent_dim) if 'past_tail' in cond else None
        self.shape_embed = TrajProcess(shape_dim, latent_dim) if cond.endswith('+shape') else None
        self.pos = PositionalEncoding(latent_dim, dropout)
        layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                                           dropout=dropout, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        if self.mode == 'query':
            self.out = nn.Linear(latent_dim, input_feats)             # one frame per output position
            self.frame_query = nn.Parameter(torch.randn(stride, latent_dim) * 0.02)
        else:
            self.out = nn.Linear(latent_dim, stride * input_feats)    # `stride` frames per token
        self.refine = TemporalRefine(input_feats, refine_layers, refine_kernel) if int(refine_layers) > 0 else None

    def forward(self, z, past_tail=None, shape_feat=None):
        bs, n, _ = z.shape
        toks = [self.z_embed(z).permute(1, 0, 2)]                    # [n, B, L]
        if self.tail_embed is not None:
            assert past_tail is not None, 'decoder_cond: past_tail needs the tail frames'
            toks.insert(0, self.tail_embed(past_tail[..., -self.n_tail:]))
        if self.shape_embed is not None:
            assert shape_feat is not None, 'decoder_cond: +shape needs shape_feat'
            toks.insert(0, self.shape_embed(shape_feat))
        if self.mode == 'query':
            # one query per output frame: token t's hidden state is no longer the only route to its `stride` frames
            zt = toks[-1]                                            # [n, B, L] the z tokens
            q = (zt.unsqueeze(1) + self.frame_query[None, :, None, :]).reshape(n * self.stride, bs, -1)
            h = self.enc(self.pos(torch.cat(toks[:-1] + [zt, q], dim=0)))[-n * self.stride:]   # [n*stride, B, L]
            o = self.out(h).reshape(n * self.stride, bs, self.njoints, self.nfeats)
            out = o.permute(1, 2, 3, 0)                              # [B, J, feat, F]
        else:
            h = self.enc(self.pos(torch.cat(toks, dim=0)))[-n:]      # [n, B, L]
            o = self.out(h).reshape(n, bs, self.stride, self.njoints, self.nfeats)
            out = o.permute(1, 3, 4, 0, 2).reshape(bs, self.njoints, self.nfeats, n * self.stride)
        if self.refine is not None:
            nf = out.shape[-1]
            r = self.refine(out.reshape(bs, self.njoints * self.nfeats, nf))
            out = r.reshape(bs, self.njoints, self.nfeats, nf)
        return out


class MotionTokenVAE(nn.Module):
    def __init__(self, input_feats, njoints, nfeats, shape_dim, d_z, stride, latent_dim, enc_layers, dec_layers,
                 num_heads, ff_size, dropout, encoder_cond='none', decoder_cond='past_tail', n_tail=5,
                 dec_mode='patch', refine_layers=0, refine_kernel=5):
        super().__init__()
        self.stride, self.d_z = stride, d_z
        self.encoder_cond, self.decoder_cond = encoder_cond, decoder_cond
        self.encoder = TokenEncoder(input_feats, d_z, stride, latent_dim, enc_layers, num_heads, ff_size, dropout,
                                    shape_dim=shape_dim if encoder_cond == 'shape' else None)
        self.decoder = TokenDecoder(input_feats, njoints, nfeats, d_z, stride, latent_dim, dec_layers, num_heads,
                                    ff_size, dropout, cond=decoder_cond, n_tail=n_tail,
                                    shape_dim=shape_dim if decoder_cond.endswith('+shape') else None,
                                    mode=dec_mode, refine_layers=refine_layers, refine_kernel=refine_kernel)

    def encode(self, x, shape_feat=None):
        return self.encoder(x, shape_feat=shape_feat)

    def encode_past(self, past, shape_feat=None):
        """[B,J,F,K] -> deterministic history token [B,1,d_z] (left edge-pad K -> stride; works on decoded frames)."""
        pad = self.stride - past.shape[-1]
        if pad > 0:
            past = torch.cat([past[..., :1].expand(*past.shape[:-1], pad), past], dim=-1)
        mu, _ = self.encoder(past[..., -self.stride:], shape_feat=shape_feat)
        return mu

    def decode(self, z, past_tail=None, shape_feat=None):
        return self.decoder(z, past_tail=past_tail, shape_feat=shape_feat)


def build_vae(cfg, meta):
    """Mirror of checkpoint.build_model's dimension computation."""
    d = cfg.diffusion
    per_rot_feat = PER_ROT_FEAT[cfg.model.rot_req]
    njoints = int(meta['joint_num']) + 2
    return MotionTokenVAE(njoints * per_rot_feat, njoints, per_rot_feat, int(meta['shape_dim']),
                          int(d.d_z), int(d.token_stride), int(d.vae_latent_dim), int(d.enc_layers),
                          int(d.dec_layers), int(d.vae_heads), int(d.vae_ff_size), float(d.vae_dropout),
                          encoder_cond=str(d.get('encoder_cond', 'none')), decoder_cond=str(d.decoder_cond),
                          n_tail=int(d.n_tail), dec_mode=str(d.get('dec_mode', 'patch')),
                          refine_layers=int(d.get('refine_layers', 0)), refine_kernel=int(d.get('refine_kernel', 5)))


def kl_free_bits(mu, logvar, free_bits):
    """KL(q(z|x) || N(0,1)) per element, clamped at free_bits nats, summed over tokens x dims, batch mean."""
    kl = 0.5 * (mu ** 2 + torch.exp(logvar) - 1.0 - logvar)
    return kl.clamp(min=float(free_bits)).sum(dim=(1, 2)).mean()


class VAEModule(MotionDiffusionModule):
    """Stage-1 codec head. `self.model` is the MotionTokenVAE (EMA/optimizer/checkpoint reload all bind to it);
    the parent-built denoiser is discarded (init-time waste, accepted to keep lit_module.py untouched).
    Run with continuity=enabled: model_output = cat(true prefix, decoded future) puts the seam terms on the
    K boundary and auto-disables loss_past_recon."""

    def __init__(self, cfg, meta):
        super().__init__(cfg, meta)
        d = cfg.diffusion
        assert str(d.type) in ('vae', 'vae_inv'), d.type
        if self.delta:
            raise SystemExit('diffusion.delta is not supported by the vae head')
        self.model = build_vae(cfg, meta)
        self.stride, self.n_future_tokens = int(d.token_stride), int(d.n_future_tokens)
        assert self.clip_len - self.past_frame == self.n_future_tokens * self.stride, \
            (self.clip_len, self.past_frame, self.n_future_tokens, self.stride)
        self.beta, self.free_bits = float(d.beta), float(d.free_bits)
        self.kl_anneal_frac = float(d.kl_anneal_frac)
        self.n_tail = int(d.n_tail)

    # ---------------------------------------------------------------- pieces
    def _beta(self):
        if self.kl_anneal_frac <= 0:
            return self.beta
        warm = max(1.0, self.kl_anneal_frac * float(self.cfg.trainer.epochs))
        return self.beta * min(1.0, float(self.current_epoch) / warm)

    def _shape_in(self, cond, which):
        d = self.cfg.diffusion
        use = (which == 'enc' and str(d.get('encoder_cond', 'none')) == 'shape') or \
              (which == 'dec' and str(d.decoder_cond).endswith('+shape'))
        return cond['shape_feat'] if use else None

    def _reconstruct(self, x0, cond, sample_z):
        K = self.past_frame
        mu, logvar = self.model.encode(x0[..., K:], shape_feat=self._shape_in(cond, 'enc'))
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps if sample_z else mu
        tail = x0[..., K - self.n_tail:K] if 'past_tail' in self.model.decoder_cond else None
        xhat_fut = self.model.decode(z, past_tail=tail, shape_feat=self._shape_in(cond, 'dec'))
        return torch.cat([x0[..., :K], xhat_fut], dim=-1), mu, logvar, eps

    # ---------------------------------------------------------------- contract
    def forward_diffuse(self, batch, **kw):
        cond = batch['conditions']
        x0 = batch['data'].permute(0, 2, 3, 1)
        batch_size = x0.shape[0]
        model_output, mu, logvar, eps = self._reconstruct(x0, cond, sample_z=self.training)
        is_clean = None
        if self.continuity.enabled:
            is_clean = x0.new_zeros(batch_size, x0.shape[-1])
            is_clean[:, :self.past_frame] = 1.0
        return {
            'model_output': model_output,
            'x0': x0,
            'x_t': x0,                                              # no corruption in this head
            't': torch.zeros(batch_size, dtype=torch.long, device=x0.device),
            'noise': eps,                                           # [B, n_tokens, d_z] (head-specific shape)
            'keep': torch.ones(batch_size, dtype=torch.bool, device=x0.device),
            'is_clean': is_clean,
            'gt_full': x0,
            'pred_full': model_output,
            'kl': kl_free_bits(mu, logvar, self.free_bits),
            'mu': mu,
            'logvar': logvar,
        }

    def _losses(self, out, cond):
        losses = super()._losses(out, cond)
        loss_kl = self._beta() * out['kl']
        losses['loss_kl'] = loss_kl
        losses['loss_kl_raw'] = out['kl'].detach()
        losses['loss'] = losses['loss'] + loss_kl
        return losses

    @torch.no_grad()
    def sample(self, batch, generator=None):
        """Deterministic reconstruction (z = mu); `generator` accepted for contract compatibility, unused."""
        cond = batch['conditions']
        x0 = batch['data'].permute(0, 2, 3, 1)
        recon, _, _, _ = self._reconstruct(x0, cond, sample_z=False)
        return x0.permute(0, 3, 1, 2), recon.permute(0, 3, 1, 2)


class VAEInvModule(VAEModule):
    """Shape-invariant-latent variant: skel/betas go into encoder AND decoder
    (encoder_cond: shape, decoder_cond: past_tail+shape) and z is pushed towards a shape-free motion
    abstraction with two hard supervisions the frame-aligned cross data uniquely affords:
      loss_zinv  -- scale-normalised MSE(mu_A, mu_B) for the same source clip on two bodies;
      loss_swap  -- decode(z_A, skel_B/tail_B) supervised against body B's REAL clip (latent retargeting).
    Trains on paired batches (data/paired_clips.py, cfg.diffusion.paired_data=true); export paths receive
    plain batches and fall through to the parent codec behaviour."""

    def __init__(self, cfg, meta):
        super().__init__(cfg, meta)
        d = cfg.diffusion
        self.w_zinv, self.w_swap = float(d.w_zinv), float(d.w_swap)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if isinstance(batch, dict) and 'a' in batch and 'b' in batch:
            from data.augment import featurize_batch
            with torch.no_grad():
                a = featurize_batch(batch['a'], self.root_pos_mean, self.root_pos_std, self.past_frame)
                b = featurize_batch(batch['b'], self.root_pos_mean, self.root_pos_std, self.past_frame)
            return {'a': a, 'b': b, 'data': a['data'], 'conditions': a['conditions']}   # aliases for harness callbacks
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def training_step(self, batch, batch_idx):
        if 'b' not in batch:
            return super().training_step(batch, batch_idx)
        a, b = batch['a'], batch['b']
        out_a = self.forward_diffuse(a)
        out_b = self.forward_diffuse(b)
        la = self._losses(out_a, a['conditions'])
        lb = self._losses(out_b, b['conditions'])
        losses = {k: la[k] + lb[k] for k in la}
        # z-invariance: same source clip, two bodies -> same latent (scale-normalised, robust to the KL scale)
        mu_a, mu_b = out_a['mu'], out_b['mu']
        zinv = torch.mean((mu_a - mu_b) ** 2) / (0.5 * torch.mean(mu_a ** 2 + mu_b ** 2) + 1e-8)
        losses['loss_zinv'] = self.w_zinv * zinv
        # swap reconstruction: a's latent + b's shape/tail must reproduce b's real clip
        K = self.past_frame
        x0b = b['data'].permute(0, 2, 3, 1)
        tail_b = x0b[..., K - self.n_tail:K] if 'past_tail' in self.model.decoder_cond else None
        xhat = self.model.decode(mu_a, past_tail=tail_b, shape_feat=self._shape_in(b['conditions'], 'dec'))
        out_swap = dict(out_b)
        out_swap['model_output'] = torch.cat([x0b[..., :K], xhat], dim=-1)
        out_swap['pred_full'] = out_swap['model_output']
        swap = MotionDiffusionModule._losses(self, out_swap, b['conditions'])           # data/geo/seam terms, no KL
        losses['loss_swap'] = self.w_swap * swap['loss']
        losses['loss'] = losses['loss'] + losses['loss_zinv'] + losses['loss_swap']
        for k, v in losses.items():
            self.log(f'train/{k}', v.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1,
                     logger=False, prog_bar=(k == 'loss'))
        return {'loss': losses['loss'], 'losses': {k: v.detach() for k, v in losses.items()}}
