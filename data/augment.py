"""Batched featurization of raw clips (GPU-side half of the data pipeline).

``featurize_batch`` applies, batched on the training device, everything that follows ``LocoDataset``'s random draws:
Y-axis rotation augmentation of the root/trajectory quaternions and of the root/trajectory xz positions,
quaternion -> 6d conversion, root normalisation, and assembly of the ``[T, J+2, 6]`` feature tensor (joint
rotations, padded root position, padded foot contact), with the ``utils.nn_transforms`` functions (CPU and GPU
results differ by FMA rounding only).
"""
import torch
import torch.nn.functional as F

from utils import nn_transforms


def rotate_xz(x, z, cos_t, sin_t):
    """Y-axis rotation of xz coordinates: x' = c*x + s*z, z' = -s*x + c*z."""
    return cos_t * x + sin_t * z, -sin_t * x + cos_t * z


def featurize_batch(raw, root_pos_mean, root_pos_std, past_frame, model_layout=True, *, end_frames=0):
    """
    raw: dict of batched tensors from ``LocoDataset`` (all on the same device):
        rotations (B,T,J,4) root_pos (B,T,3) foot_contact (B,T,2) traj_quat (B,F,4) traj_xz (B,F,2)
        aug_trig (B,4) = [cos(theta/2), sin(theta/2), cos(theta), sin(theta)], text_feat (B,512),
        shape_feat (B,S), skel_offset (B,J,3), motion_idx (B,), optionally style_idx (B,) (LocoDataset clips;
        hand-built raw batches such as realtime/model_loop.py omit it)
    root_pos_mean/std: (3,) tensors on the same device.
    end_frames: >0 = the in-betweening switch -- also return the LAST ``end_frames`` real frames of the window
        under ``conditions['end_motion']`` so the stage-2 prior can build its end token from them.  0 (the
        default) adds no key and leaves every tensor untouched.

    Returns {'data': (B,T,J+2,6), 'data_delta': same, 'conditions': {...}}.
    With ``model_layout=True`` the conditions are permuted to the model layout (as network/diffusion.py prepare_cond:
    past_motion -> (B,J+2,6,K), traj -> (B,C,F), shape/text -> (B,D,1)); with ``False`` they keep the per-item
    dataset layout.
    """
    c2, s2, c, s = raw['aug_trig'].unbind(-1)                              # (B,)
    zero = torch.zeros_like(c2)
    q_aug = torch.stack([c2, zero, s2, zero], dim=-1).unsqueeze(1)          # (B,1,4)

    rotations = raw['rotations'].clone()                                    # (B,T,J,4)
    rotations[:, :, 0] = nn_transforms.quat_mul(q_aug, rotations[:, :, 0])
    traj_quat = nn_transforms.quat_mul(q_aug, raw['traj_quat'])             # (B,F,4)

    c_, s_ = c.unsqueeze(1), s.unsqueeze(1)                                 # (B,1)
    rx, ry, rz = raw['root_pos'].unbind(-1)
    rx, rz = rotate_xz(rx, rz, c_, s_)
    root_pos = torch.stack([rx, ry, rz], dim=-1)                            # (B,T,3)
    tx, tz = raw['traj_xz'].unbind(-1)
    tx, tz = rotate_xz(tx, tz, c_, s_)
    traj_xz = torch.stack([tx, tz], dim=-1)                                 # (B,F,2)

    rot6 = nn_transforms.quat2repr6d(rotations)                             # (B,T,J,6)
    traj6 = nn_transforms.quat2repr6d(traj_quat)                            # (B,F,6)

    root_norm = (root_pos - root_pos_mean) / root_pos_std
    per_rot_feat = rot6.shape[-1]
    root_pad = F.pad(root_norm, (0, per_rot_feat - 3)).unsqueeze(2)         # (B,T,1,6)
    fc_pad = F.pad(raw['foot_contact'], (0, per_rot_feat - 2)).unsqueeze(2)  # (B,T,1,6)
    data = torch.cat([rot6, root_pad, fc_pad], dim=2)                       # (B,T,J+2,6)

    past = data[:, :past_frame].clone()
    past[:, :, -1] = 0
    delta = data - past[:, -1:]
    mask = torch.ones(data.shape[0], data.shape[1], dtype=torch.bool, device=data.device)

    cond = {
        'past_motion': past,
        'traj_pose': traj6,
        'traj_trans': traj_xz,
        'shape_feat': raw['shape_feat'],
        'text_feat': raw['text_feat'],
        'skel_offset': raw['skel_offset'],
        'mask': mask,
        'motion_idx': raw['motion_idx'],
        'clip_start': raw['clip_start'],
    }
    if 'style_idx' in raw:
        cond['style_idx'] = raw['style_idx']                                # (B,) rows of STYLE_VOCAB, layout-free
    for k in ('subject_idx', 'attr_idx'):                                  # (B,) / (B,3) persona rows, layout-free
        if k in raw:
            cond[k] = raw[k]
    if 'audio_feat' in raw:                                                  # speech gestures: (B, Fa, D) per-frame audio
        cond['audio_feat'] = raw['audio_feat']
    if end_frames:
        cond['end_motion'] = data[:, -int(end_frames):].clone()            # (B, n, J+2, 6) the window's last n frames
    if model_layout:
        cond['past_motion'] = cond['past_motion'].permute(0, 2, 3, 1)
        cond['traj_pose'] = cond['traj_pose'].permute(0, 2, 1)
        cond['traj_trans'] = cond['traj_trans'].permute(0, 2, 1)
        cond['shape_feat'] = cond['shape_feat'].unsqueeze(-1)
        cond['text_feat'] = cond['text_feat'].unsqueeze(-1)
        if 'end_motion' in cond:
            cond['end_motion'] = cond['end_motion'].permute(0, 2, 3, 1)    # (B, J+2, 6, n), like past_motion
        if 'audio_feat' in cond:
            cond['audio_feat'] = cond['audio_feat'].permute(0, 2, 1)       # (B, D, Fa), like traj_pose
    return {'data': data, 'data_delta': delta, 'conditions': cond}
