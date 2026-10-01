"""Model/scheduler construction from a Hydra config + dataset meta, and weight loading for both checkpoint formats."""
import numpy as np
import torch
from diffusers import DDPMScheduler

from network.models import MotionSkelDiffusionMLP

PER_ROT_FEAT = {'q': 4, '6d': 6, 'euler': 3}


def public_meta(meta):
    """The pickle-free part of a dataset meta.pkl (saved as Lightning hyper-parameters).  Dataset-provided vocabularies
    and the audio feature width (speech-driven gesture application) are added only when the dataset has them, so
    every locomotion checkpoint's hyper-parameters are unchanged."""
    out = {
        'joint_num': int(meta['joint_num']),
        'shape_dim': int(meta['shape_dim']),
        'parents': [int(p) for p in meta['parents']],
        'names': list(meta['names']),
        'root_pos_mean': np.asarray(meta['root_pos_mean'], dtype=np.float32).tolist(),
        'root_pos_std': np.asarray(meta['root_pos_std'], dtype=np.float32).tolist(),
    }
    for k in ('subject_vocab', 'style_vocab'):
        if meta.get(k):
            out[k] = [str(v) for v in meta[k]]
    if meta.get('audio_dim'):
        out['audio_dim'] = int(meta['audio_dim'])
    return out


def build_model(cfg, meta):
    per_rot_feat = PER_ROT_FEAT[cfg.model.rot_req]
    njoints = int(meta['joint_num']) + 2
    clip_len = int(cfg.data.past_frame) + int(cfg.data.future_frame)
    cond_mask_prob = 0.0 if cfg.continuity.enabled else float(cfg.model.cond_mask_prob)
    return MotionSkelDiffusionMLP(
        njoints * per_rot_feat, int(meta['shape_dim']), njoints, per_rot_feat,
        cfg.model.rot_req, clip_len,
        cfg.model.latent_dim, cfg.model.ff_size, cfg.model.num_layers, cfg.model.num_heads,
        dropout=cfg.model.dropout, cond_mask_prob=cond_mask_prob,
        # structured conditions: defaults 'none'/'text' = one CLIP token of a fused text prompt (default off)
        style_cond=str(cfg.model.get('style_cond', 'none')), persona_cond=str(cfg.model.get('persona_cond', 'text')),
    )


def build_scheduler(diffusion_cfg):
    return DDPMScheduler(
        num_train_timesteps=int(diffusion_cfg.num_train_timesteps),
        beta_schedule=diffusion_cfg.beta_schedule,
        prediction_type=diffusion_cfg.prediction_type,
        variance_type=diffusion_cfg.variance_type,
        clip_sample=bool(diffusion_cfg.clip_sample),
    )


def extract_model_state(checkpoint):
    """Raw-model state_dict from a plain {'state_dict': ...} .pt or a Lightning .ckpt."""
    sd = checkpoint['state_dict']
    if any(k.startswith('model.') for k in sd):
        sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    return sd


def fill_released(sd, own):
    """Released checkpoints store the weights in fp16 (load_state_dict casts them back to fp32) and leave out the
    fixed sinusoid tables (`*.pe`, rebuilt by the module's constructor): take those from the fresh module."""
    for k, v in own.items():
        if k not in sd and k.endswith('.pe'):
            sd[k] = v
    return sd


def ema_merged(checkpoint):
    """True when the checkpoint's state_dict already holds the EMA weights (released checkpoints)."""
    return checkpoint.get('weights') == 'ema'


def load_weights(model, path, map_location='cpu'):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    sd = extract_model_state(checkpoint)
    own = model.state_dict()
    fill_released(sd, own)
    if 'clean_token' in own and 'clean_token' not in sd:
        sd['clean_token'] = torch.zeros_like(own['clean_token'])
    model.load_state_dict(sd, strict=True)
    return checkpoint
