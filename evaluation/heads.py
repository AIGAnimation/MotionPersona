"""Checkpoint bundles and sampling heads for the rollout engine.

A head turns one featurized batch + caller-drawn noise into a [B, K+F, J+2, 6] sample; the engine owns the noise so
that every case's draws are a pure function of (seed, case, block) -- see evaluation/engine.py.
"""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from data.loco_dataset import read_meta
from network.checkpoint import ema_merged, extract_model_state, fill_released
from network.diffusion import ddpm_sample, de_delta
from network.factory import build_module

REPO = Path(__file__).resolve().parents[1]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Bundle:
    module: object
    cfg: object
    meta: dict
    T_pose: object
    kind: str                 # latfm | ddpm | vae
    K: int
    F: int
    J: int
    device: str
    ckpt: str
    sha256: dict = field(default_factory=dict)
    root_basis: str = 'joint'
    ema: bool = True

    @property
    def clip_len(self):
        return self.K + self.F


def load_bundle(run_or_ckpt, device='cpu', ema=True, steps=None, sampler=None, hist_guidance=1.0, tpose_dir=None):
    """Load a run's `best.ckpt` (or a checkpoint path) the way realtime/model_loop.py::PersonaModel does: strict
    state_dict, EMA shadow weights, the codec resolved from the prior's own cfg."""
    p = Path(run_or_ckpt)
    ckpt = p / 'best.ckpt' if p.is_dir() else p
    if not ckpt.is_absolute() and not ckpt.exists():
        ckpt = REPO / ckpt
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(state['hyper_parameters']['cfg'])
    meta = state['hyper_parameters']['meta']
    kind = str(cfg.diffusion.get('type', 'ddpm'))
    shas = {'ckpt': sha256_of(ckpt)}
    if kind == 'latfm':
        vae = Path(str(cfg.diffusion.vae_ckpt))
        if not vae.is_absolute():                                            # released ckpts: codec next to the prior
            vae = ckpt.parent / vae if (ckpt.parent / vae).exists() else REPO / vae
        cfg.diffusion.vae_ckpt = str(vae)
        from network.latfm import resolve_vae_ckpt
        shas['codec'] = sha256_of(resolve_vae_ckpt(str(vae)))
    module = build_module(cfg, meta)
    sd = extract_model_state(state)
    own = module.model.state_dict()
    fill_released(sd, own)
    if 'clean_token' in own and 'clean_token' not in sd:
        sd['clean_token'] = torch.zeros_like(own['clean_token'])
    module.model.load_state_dict(sd, strict=True)
    if ema and not ema_merged(state):                                   # released ckpts: already EMA
        from torch_ema import ExponentialMovingAverage
        ema_state = state.get('callbacks', {}).get('EMACallback')
        if not ema_state:
            raise SystemExit(f'{ckpt}: no EMA weights stored (the evaluation uses the EMA weights)')
        shadow = ExponentialMovingAverage(module.model.parameters(), decay=ema_state['decay'])
        shadow.load_state_dict(ema_state)
        shadow.copy_to()
    module.eval().to(device)
    module.hist_guidance = float(hist_guidance)
    if kind in ('latfm', 'fm'):
        module.fm_steps = int(steps) if steps else 2                     # the protocol: two Euler steps
        module.fm_sampler = str(sampler) if sampler else 'euler'
    elif steps:
        cfg.diffusion.num_inference_steps = int(steps)
    # the BVH template lives in the dataset's meta.pkl (not in the checkpoint); every dataset shares the repo skeleton,
    # so a machine without the training data takes it from data/skeleton
    dp = Path(str(cfg.data.path)); dp = dp if dp.is_absolute() else REPO / dp
    if tpose_dir is not None:
        dp = Path(tpose_dir)
    elif not (dp / 'meta.pkl').exists():
        dp = REPO / 'data/skeleton'
    T_pose = read_meta(dp)['T_pose']
    return Bundle(module=module, cfg=cfg, meta=meta, T_pose=T_pose, kind=kind, K=int(cfg.data.past_frame),
                  F=int(cfg.data.future_frame), J=int(meta['joint_num']), device=device, ckpt=str(ckpt), sha256=shas,
                  root_basis=str(cfg.data.get('skel_root_basis', 'joint')), ema=ema)


class LatFMHead:
    """Two-stage model: few-step Euler in the codec's token space + one decode. One noise draw per block."""

    def __init__(self, bundle, shape_split=None):
        self.b = bundle
        m = bundle.module
        self.n_tokens, self.d_z = int(m.n_tokens), int(m.d_z)
        self.steps, self.sampler = int(m.fm_steps), str(m.fm_sampler)
        self.shape_split = shape_split                 # None | ('prior'|'decoder'|'both', base_betas tensor (S,))

    def noise_shapes(self):
        return [(self.n_tokens, self.d_z)]

    @torch.no_grad()
    def sample(self, batch, noise):
        kw = {}
        if self.shape_split is not None:
            mode, base = self.shape_split
            B = batch['conditions']['shape_feat'].shape[0]
            base_b = base.to(batch['conditions']['shape_feat']).view(1, -1, 1).expand(B, -1, 1)
            # 'prior': the prior sees the target body, the decoder the base body; 'decoder': the reverse; 'both': target everywhere
            if mode == 'prior':
                kw['shape_dec'] = base_b
            elif mode == 'decoder':
                kw['shape_prior'] = base_b
        _, out = self.b.module.sample(batch, x_init=noise[0], **kw)
        return out


class DDPMHead:
    """Raw-space DDPM baseline: ancestral sampling with the history prefix pinned, one initial
    draw plus one per ancestral step."""

    def __init__(self, bundle):
        self.b = bundle
        m = bundle.module
        self.steps = int(m.cfg.diffusion.num_inference_steps or m.scheduler.config.num_train_timesteps)
        self.shape = (bundle.J + 2, 6, bundle.clip_len)

    def noise_shapes(self):
        return [self.shape] * (1 + self.steps)

    @torch.no_grad()
    def sample(self, batch, noise):
        m = self.b.module
        cond = batch['conditions']
        x = ddpm_sample(m.model, m.scheduler, cond, past_frame=m.past_frame, clip_len=m.clip_len, delta=m.delta,
                        continuity_enabled=m.continuity.enabled, num_inference_steps=self.steps,
                        guide=m.guided_by_history if m.hist_guidance != 1.0 else None, noise_fn=lambda i: noise[i])
        return de_delta(x, cond['past_motion'], m.delta).permute(0, 3, 1, 2)


class FMHead:
    """Raw-space rectified flow: Euler in pose space with the history prefix re-pinned; one draw."""

    def __init__(self, bundle):
        self.b = bundle
        m = bundle.module
        self.steps, self.sampler = int(m.fm_steps), str(m.fm_sampler)
        self.shape = (bundle.J + 2, 6, bundle.clip_len)

    def noise_shapes(self):
        return [self.shape]

    @torch.no_grad()
    def sample(self, batch, noise):
        _, out = self.b.module.sample(batch, x_init=noise[0])
        return out


def make_head(bundle, shape_split=None):
    if bundle.kind == 'latfm':
        return LatFMHead(bundle, shape_split=shape_split)
    if bundle.kind == 'ddpm':
        return DDPMHead(bundle)
    if bundle.kind == 'fm':
        return FMHead(bundle)
    raise SystemExit(f'no rollout head for diffusion.type={bundle.kind}')
