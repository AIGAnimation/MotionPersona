"""Diffusion forward/sampling core, free of Lightning so tests, export and the training module share one path.

Layout convention: motion tensors are ``[B, J+2, F, T]`` (time last) — the model's native layout; datasets
produce ``[B, T, J+2, F]`` and are permuted here.
"""
from dataclasses import dataclass

import torch
from network.cond_tokens import cond_token_kwargs


@dataclass(frozen=True)
class ContinuityCfg:
    enabled: bool = False
    window: int = 5
    w_pos: float = 0.5
    w_vel: float = 0.5
    w_acc: float = 0.25
    use_xyz: bool = True


def prepare_cond(cond):
    """Conditions from dataset layout to model layout, without mutating the input dict."""
    out = dict(cond)
    out['past_motion'] = cond['past_motion'].permute(0, 2, 3, 1)
    out['traj_pose'] = cond['traj_pose'].permute(0, 2, 1)
    out['traj_trans'] = cond['traj_trans'].permute(0, 2, 1)
    out['shape_feat'] = cond['shape_feat'].unsqueeze(-1)
    out['text_feat'] = cond['text_feat'].unsqueeze(-1)
    return out


def prefix_from_past(past_motion, delta):
    """The clean prefix in training space, derived from what the consumer has (``past_motion``)."""
    return past_motion - past_motion[..., -1:] if delta else past_motion


def de_delta(x, past_motion, delta):
    return x + past_motion[..., -1:] if delta else x


def diffusion_forward(model, scheduler, x_start, cond, *, past_frame, delta, cond_mask_prob, continuity_enabled,
                      noise=None, t=None, keep=None):
    """One noising + denoising pass, RNG order: randn, randint, rand.

    x_start: [B, T, J, F] (dataset layout). cond: model layout. Returns a dict with everything in [B, J, F, T].
    """
    x0 = x_start.permute(0, 2, 3, 1)
    batch_size = x0.shape[0]
    device = x0.device

    if noise is None:
        noise = torch.randn_like(x0)
    if t is None:
        t = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
    x_t = scheduler.add_noise(x0, noise, t)

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
        is_clean = x0.new_zeros(batch_size, x0.shape[-1])
        is_clean[:, :past_frame] = 1.0

    model_output = model(x_t, t, past_in, cond['traj_pose'], cond['traj_trans'], cond['shape_feat'], cond['text_feat'], is_clean,
                         **cond_token_kwargs(model, cond))

    return {
        'model_output': model_output,
        'x0': x0,
        'x_t': x_t,
        't': t,
        'noise': noise,
        'keep': keep,
        'is_clean': is_clean,
        'gt_full': de_delta(x0, past_motion, delta),
        'pred_full': de_delta(model_output, past_motion, delta),
    }


def posterior_coefficients(scheduler):
    """DDPM posterior coefficients of ``DDPMScheduler.step`` (variance_type=fixed_small), exported for the ONNX consumer.

    x_{t-1} = mean_coef1[t] * x0_pred + mean_coef2[t] * x_t + [t > 0] * exp(0.5 * log_variance[t]) * eps
    """
    betas = scheduler.betas
    alphas = scheduler.alphas
    alphas_cumprod = scheduler.alphas_cumprod
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        'posterior_log_variance_clipped': torch.log(posterior_variance.clamp(min=1e-20)),
        'posterior_mean_coef1': betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        'posterior_mean_coef2': (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
    }


@torch.no_grad()
def ddpm_sample(model, scheduler, cond, *, past_frame, clip_len, delta, continuity_enabled,
                num_inference_steps=None, generator=None, guide=None, noise_fn=None):
    """DDPM ancestral sampling with the prefix re-pinned after every step (the loop the ONNX consumer mirrors).

    Returns the sample in *training space* ([B, J, F, T]); apply ``de_delta`` for the raw motion.
    ``noise_fn(i)`` -> tensor of the sample's shape supplies the initial noise (i = 0) and the ancestral noise of
    step i >= 1 in place of the generator; the step then uses ``posterior_coefficients`` (fixed_small variance) so the
    evaluation engine can draw every case's noise on its own (batched == serial). None keeps the scheduler path.
    """
    past_motion = cond['past_motion']
    batch_size, njoints, nfeats, _ = past_motion.shape
    device = past_motion.device
    shape = (batch_size, njoints, nfeats, clip_len)
    if noise_fn is not None:
        x = noise_fn(0).to(device).clone()
        assert tuple(x.shape) == shape, (x.shape, shape)
    elif generator is not None and generator.device.type != device.type:
        x = torch.randn(shape, generator=generator).to(device)
    else:
        x = torch.randn(shape, generator=generator, device=device)

    is_clean, prefix = None, None
    if continuity_enabled:
        prefix = prefix_from_past(past_motion, delta)
        x[..., :past_frame] = prefix
        is_clean = x.new_zeros(batch_size, clip_len)
        is_clean[:, :past_frame] = 1.0

    scheduler.set_timesteps(num_inference_steps or scheduler.config.num_train_timesteps, device=device)
    coef = posterior_coefficients(scheduler) if noise_fn is not None else None
    for i, t in enumerate(scheduler.timesteps):
        t_batch = t.reshape(1).expand(batch_size).to(device)
        kw = cond_token_kwargs(model, cond)
        call = lambda h: model(x, t_batch, h, cond['traj_pose'], cond['traj_trans'], cond['shape_feat'], cond['text_feat'], is_clean, **kw)  # noqa: E731
        x0_pred = call(past_motion) if guide is None else guide(call, past_motion)
        if noise_fn is None:
            x = scheduler.step(x0_pred, t, x, generator=generator).prev_sample
        else:                                   # DDPMScheduler.step, variance_type fixed_small, with our own noise
            ti = int(t)
            mean = coef['posterior_mean_coef1'][ti] * x0_pred + coef['posterior_mean_coef2'][ti] * x
            if ti > 0:
                sigma = torch.exp(0.5 * coef['posterior_log_variance_clipped'][ti])
                x = mean + sigma * noise_fn(i + 1).to(device)
            else:
                x = mean
        if continuity_enabled:
            x[..., :past_frame] = prefix
    return x
