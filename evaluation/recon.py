"""Codec teacher-forced reconstruction of reference takes:
the take on a body is cut into 45-frame blocks at t = K + 45 k; each block's history is the TRUE K frames before it
and the encoder sees the TRUE 45 frames, the decoder renders them back (z = mu, deterministic). Output = frames
K .. K + 45 n of the take (the first K frames are the untouched history), with the decoded contact channel as the
reconstruction's contact flag (reference-label IoU compares it with the source labels).
"""
from dataclasses import dataclass

import numpy as np
import torch

from utils.nn_transforms import repr6d2quat

TEXT_DIM = 512


@dataclass
class ReconResult:
    quats: np.ndarray
    root: np.ndarray
    contact: np.ndarray
    label: np.ndarray           # the source labels on the same frames
    frame0: int                 # the take frame of output frame 0 (= K)
    n_blocks: int
    frame_of_block: np.ndarray


def reconstruct_take(bundle, z, *, batch_size=64, device=None):
    """bundle = a codec bundle (evaluation.heads.load_bundle on a vae run); z = a reference row npz (rotations, root_pos,
    foot_contact, skel_offset, betas)."""
    dev = device or bundle.device
    K, F, J = bundle.K, bundle.F, bundle.J
    rot, root, fc = np.asarray(z['rotations'], np.float32), np.asarray(z['root_pos'], np.float32), np.asarray(z['foot_contact'], np.float32)
    T = len(rot)
    n = (T - K) // F
    if n < 1:
        return None
    m = bundle.module
    starts = [K + F * k for k in range(n)]
    out_q = np.zeros((n * F, J, 4), np.float32); out_r = np.zeros((n * F, 3), np.float32); out_c = np.zeros((n * F, 2), np.float32)
    for b0 in range(0, n, batch_size):
        ss = starts[b0:b0 + batch_size]; B = len(ss)
        rots = np.stack([rot[s - K:s + F] for s in ss]); roots = np.stack([root[s - K:s + F] for s in ss]).copy(); fcs = np.stack([fc[s - K:s + F] for s in ss])
        pivot = roots[:, K - 1, [0, 2]].copy()
        roots[:, :, [0, 2]] -= pivot[:, None, :]
        raw = {'rotations': torch.tensor(rots, device=dev), 'root_pos': torch.tensor(roots, device=dev), 'foot_contact': torch.tensor(fcs, device=dev),
               'traj_quat': torch.zeros(B, F, 4, device=dev), 'traj_xz': torch.zeros(B, F, 2, device=dev),
               'aug_trig': torch.tensor([[1.0, 0.0, 1.0, 0.0]] * B, device=dev),
               'text_feat': torch.zeros(B, TEXT_DIM, device=dev), 'shape_feat': torch.tensor(np.tile(np.asarray(z['betas'], np.float32), (B, 1)), device=dev),
               'skel_offset': torch.tensor(np.tile(np.asarray(z['skel_offset'], np.float32), (B, 1, 1)), device=dev),
               'motion_idx': torch.zeros(B, dtype=torch.int64, device=dev), 'clip_start': torch.zeros(B, dtype=torch.int64, device=dev)}
        raw['traj_quat'][:, :, 0] = 1.0                                       # identity yaw (the codec ignores the trajectory)
        batch = m.featurize(raw)
        with torch.no_grad():
            _, rec = m.sample(batch)                                          # (B, K+F, J+2, 6), z = mu
        fut = rec[:, K:]
        q = repr6d2quat(fut[:, :, :J]).cpu().numpy(); r = (fut[:, :, J, :3] * m.root_pos_std + m.root_pos_mean).cpu().numpy()
        r[:, :, [0, 2]] += pivot[:, None, :]
        c = fut[:, :, J + 1, :2].cpu().numpy()
        for i, s in enumerate(ss):
            o = s - K
            out_q[o:o + F] = q[i]; out_r[o:o + F] = r[i]; out_c[o:o + F] = c[i]
    return ReconResult(quats=out_q, root=out_r, contact=out_c, label=fc[K:K + n * F], frame0=K, n_blocks=n, frame_of_block=np.arange(n) * F)
