"""Rectified-flow head in motion space.

Math: data endpoint x1 at t=1, noise at t=0; x_t = (1-t)*noise + t*x1; the model predicts the constant
velocity v = x1 - noise; x1_hat = x_t + (1-t)*v_hat feeds the UNCHANGED MotionDiffusionModule loss stack
(geo/contact/seam supervision carries over; loss_data on x1_hat is an implicit (1-t)^2-weighted v-MSE).
Training t is drawn at integer-bin centres t = (idx + 0.5) / time_bins so the discrete TimestepEmbedder
(pe-table lookup, network/models.py) is reused unchanged and (1-t) never hits 0.

`fm_forward` mirrors network/diffusion.py::diffusion_forward line-for-line (same RNG order randn -> randint
-> rand, same keep-mask and continuity blocks, same 9 return keys + v extras) so gt/masking semantics are
identical to the DDPM head; `fm_sample_loop` is shape-agnostic and is reused by the latent head.
"""
import torch

from network.diffusion import prefix_from_past, de_delta
from network.lit_module import MotionDiffusionModule
from network.cond_tokens import cond_token_kwargs


# ------------------------------------------------------------------ core (shape-agnostic; t broadcasts)
def t_broadcast(t_cont, x):
    return t_cont.view(-1, *([1] * (x.ndim - 1)))


def fm_bin_to_t(t_idx, time_bins):
    """Bin index -> continuous t at the bin centre (never exactly 0 or 1)."""
    return (t_idx.float() + 0.5) / time_bins


def fm_interpolate(x1, noise, t_cont):
    t = t_broadcast(t_cont, x1)
    return (1.0 - t) * noise + t * x1


def fm_x1_from_v(x_t, v, t_cont):
    t = t_broadcast(t_cont, x_t)
    return x_t + (1.0 - t) * v


def fm_step_bin(t, time_bins):
    return min(int(t * time_bins), time_bins - 1)


@torch.no_grad()
def fm_sample_loop(v_fn, shape, *, time_bins, steps, sampler='euler', generator=None, device='cpu', pin_fn=None, x_init=None):
    """Integrate dx/dt = v from t=0 (noise) to t=1 (data); Euler or Heun. v_fn(x, t_idx [B] long) -> v.
    pin_fn(x) -> x re-pins the clean prefix after every update (None: no pinning, e.g. latent space).
    Initial noise follows ddpm_sample's generator semantics (CPU generator -> randn on CPU, then .to(device)), or is
    the caller's own tensor `x_init` (the evaluation engine draws it per case so a batch equals the serial run)."""
    if x_init is not None:
        assert tuple(x_init.shape) == tuple(shape), (x_init.shape, shape)
        x = x_init.to(device).clone()
    elif generator is not None and generator.device.type != torch.device(device).type:
        x = torch.randn(shape, generator=generator).to(device)
    else:
        x = torch.randn(shape, generator=generator, device=device)
    if pin_fn is not None:
        x = pin_fn(x)
    x_init = x.clone()                    # the 'x1' sampler re-interpolates along the line from this noise
    batch_size = shape[0]
    dt = 1.0 / steps
    for k in range(steps):
        t_now = k * dt
        idx = torch.full((batch_size,), fm_step_bin(t_now, time_bins), dtype=torch.long, device=x.device)
        v1 = v_fn(x, idx)
        if sampler == 'x1':
            # x1-anchored step (DDPM-x0-style, non-compounding): predict x1, re-interpolate towards it
            x1_hat = x + (1.0 - t_now) * v1
            t_next = (k + 1) * dt
            x = t_next * x1_hat + (1.0 - t_next) * x_init
        elif sampler == 'heun':
            x_e = x + dt * v1
            if pin_fn is not None:
                x_e = pin_fn(x_e)
            idx2 = torch.full((batch_size,), fm_step_bin(t_now + dt, time_bins), dtype=torch.long, device=x.device)
            x = x + dt * 0.5 * (v1 + v_fn(x_e, idx2))
        else:
            x = x + dt * v1
        if pin_fn is not None:
            x = pin_fn(x)
    return x


def fm_forward(model, x_start, cond, *, time_bins, past_frame, delta, cond_mask_prob, continuity_enabled,
               noise=None, t_idx=None, keep=None):
    """One FM corruption + prediction pass; mirror of diffusion_forward (RNG order: randn, randint, rand).

    x_start: [B, T, J, F] (dataset layout). cond: model layout. Returns the diffusion_forward contract
    ([B, J, F, T], keys model_output/x0/x_t/t/noise/keep/is_clean/gt_full/pred_full) + v_pred/v_target/t_cont.
    """
    x1 = x_start.permute(0, 2, 3, 1)
    batch_size = x1.shape[0]
    device = x1.device

    if noise is None:
        noise = torch.randn_like(x1)
    if t_idx is None:
        t_idx = torch.randint(0, time_bins, (batch_size,), device=device)
    t_cont = fm_bin_to_t(t_idx, time_bins)
    x_t = fm_interpolate(x1, noise, t_cont)

    past_motion = cond['past_motion']
    if keep is None:
        if model.training and cond_mask_prob > 0:
            keep = torch.rand(batch_size, device=past_motion.device) < (1 - cond_mask_prob)
        else:
            keep = torch.ones(batch_size, dtype=torch.bool, device=past_motion.device)
    past_in = past_motion * keep.view((batch_size, 1, 1, 1))

    is_clean = None
    if continuity_enabled:
        prefix = prefix_from_past(past_motion, delta)
        x_t = x_t.clone()
        x_t[..., :past_frame] = prefix
        is_clean = x1.new_zeros(batch_size, x1.shape[-1])
        is_clean[:, :past_frame] = 1.0

    v_pred = model(x_t, t_idx, past_in, cond['traj_pose'], cond['traj_trans'], cond['shape_feat'], cond['text_feat'], is_clean,
                   **cond_token_kwargs(model, cond))
    x1_hat = fm_x1_from_v(x_t, v_pred, t_cont)

    return {
        'model_output': x1_hat,
        'x0': x1,
        'x_t': x_t,
        't': t_idx,
        'noise': noise,
        'keep': keep,
        'is_clean': is_clean,
        'gt_full': de_delta(x1, past_motion, delta),
        'pred_full': de_delta(x1_hat, past_motion, delta),
        'v_pred': v_pred,
        'v_target': x1 - noise,
        't_cont': t_cont,
    }


# ------------------------------------------------------------------ module
class FlowMatchingModule(MotionDiffusionModule):
    """Drop-in head: same losses/callbacks/export contracts as the DDPM parent; only the corruption and the
    sampler differ. The DDPMScheduler the parent builds (from the compat-shim keys in configs/diffusion/fm.yaml)
    exists but is never used. Export semantics: 'pred' = one-pass x1_hat at a random t, 'sample' = the ODE rollout."""

    def __init__(self, cfg, meta):
        super().__init__(cfg, meta)
        d = cfg.diffusion
        assert str(d.type) == 'fm', d.type
        if self.delta:
            raise SystemExit('diffusion.delta is not supported by the fm head')
        self.time_bins = int(d.time_bins)
        self.fm_steps = int(d.sample_steps)
        self.fm_sampler = str(d.sampler)
        self.v_loss_w = float(d.get('v_loss_w', 0.0))
        pe_len = self.model.embed_timestep.sequence_pos_encoder.pe.shape[0]
        assert self.time_bins <= pe_len, f'time_bins {self.time_bins} exceeds the embedder table ({pe_len})'

    def forward_diffuse(self, batch, **kw):
        return fm_forward(self.model, batch['data'], batch['conditions'], time_bins=self.time_bins,
                          past_frame=self.past_frame, delta=self.delta, cond_mask_prob=self.cond_mask_prob,
                          continuity_enabled=self.continuity.enabled, **kw)

    def _losses(self, out, cond):
        losses = super()._losses(out, cond)
        if self.v_loss_w > 0:
            K = self.past_frame
            loss_v = self.v_loss_w * torch.mean((out['v_pred'][..., K:] - out['v_target'][..., K:]) ** 2)
            losses['loss_v'] = loss_v
            losses['loss'] = losses['loss'] + loss_v
        return losses

    @torch.no_grad()
    def sample(self, batch, generator=None, *, x_init=None):
        """ODE sampling (prefix re-pinned every update under continuity). Returns (gt_full, sample) in [B,T,J,F].
        `x_init` replaces the generator's initial noise (the evaluation engine draws it per case)."""
        cond = batch['conditions']
        kw = cond_token_kwargs(self.model, cond)
        past_motion = cond['past_motion']
        batch_size, njoints, nfeats, _ = past_motion.shape
        shape = (batch_size, njoints, nfeats, self.clip_len)
        is_clean, pin_fn = None, None
        if self.continuity.enabled:
            prefix = prefix_from_past(past_motion, self.delta)
            is_clean = past_motion.new_zeros(batch_size, self.clip_len)
            is_clean[:, :self.past_frame] = 1.0

            def pin_fn(x, _prefix=prefix, _k=self.past_frame):
                x[..., :_k] = _prefix
                return x

        def v_fn(x, t_idx):
            return self.guided_by_history(
                lambda h: self.model(x, t_idx, h, cond['traj_pose'], cond['traj_trans'],
                                     cond['shape_feat'], cond['text_feat'], is_clean, **kw), past_motion)

        x = fm_sample_loop(v_fn, shape, time_bins=self.time_bins, steps=self.fm_steps, sampler=self.fm_sampler,
                           generator=generator, device=past_motion.device, pin_fn=pin_fn, x_init=x_init)
        gt_full = de_delta(batch['data'].permute(0, 2, 3, 1), past_motion, self.delta)
        return gt_full.permute(0, 3, 1, 2), de_delta(x, past_motion, self.delta).permute(0, 3, 1, 2)
