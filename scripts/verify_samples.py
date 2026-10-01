"""Gate G6: verify exported BVH samples and compute seam metrics.

Importable (``TrainMonitor`` calls ``seam_metrics`` / ``check_bvh_dir`` after every export and writes
``samples/<tag>/summary.json``) and usable as a CLI on an existing samples directory::

    python scripts/verify_samples.py save/<run>/samples/epoch10 --frames 55
"""
import sys
sys.path.append('./')

import glob
import json
import os
import argparse

import numpy as np
import torch

from utils import nn_transforms


def pinned(gt, pred, past_frame):
    """The sequence the consumer actually plays: true history followed by the generated future. [B,T,J,F]"""
    return torch.cat([gt[:, :past_frame], pred[:, past_frame:]], dim=1)


@torch.no_grad()
def seam_metrics(motion_feat, skeletons, parents, root_pos_mean, root_pos_std, past_frame, rot_req='6d'):
    """Joint-position continuity statistics (metres) of a [B,T,J+2,6] feature tensor around frame ``past_frame``.

    seam_pos_jump = |x_K - x_{K-1}|  (the seam velocity), seam_vel_jump = |(x_K - x_{K-1}) - (x_{K-1} - x_{K-2})|
    (the seam acceleration), both averaged over batch and joints; mean_* are whole-clip averages for reference.
    """
    K = past_frame
    rot = motion_feat[:, :, :-2]
    root = motion_feat[:, :, -2, :3] * root_pos_std + root_pos_mean
    xyz = nn_transforms.neural_FK(rot, skeletons, root, parents[None], rotation_type=rot_req)   # [B,T,J,3]
    v = xyz[:, 1:] - xyz[:, :-1]
    a = v[:, 1:] - v[:, :-1]
    j = a[:, 1:] - a[:, :-1]

    def mean_norm(t):
        return float(t.norm(dim=-1).mean())

    return {
        'seam_pos_jump': mean_norm(v[:, K - 1]),
        'seam_vel_jump': mean_norm(a[:, K - 2]),
        'seam_acc_jump': mean_norm(j[:, K - 3]) if K >= 3 else None,
        'mean_vel': mean_norm(v),
        'mean_acc': mean_norm(a),
        'mean_jerk': mean_norm(j),
        'future_mean_vel': mean_norm(v[:, K - 1:]),
        'finite': bool(torch.isfinite(xyz).all()),
    }


def check_bvh_dir(sample_dir, expect_frames=None):
    """Load every BVH back and check frame count / finiteness. Returns a JSON-able report."""
    from utils.bvh_motion import Motion
    files = sorted(glob.glob(os.path.join(sample_dir, '*.bvh')))
    problems = []
    for f in files:
        try:
            m = Motion.load_bvh(f)
            n = int(m.rotations.shape[0])
            if expect_frames is not None and n != expect_frames:
                problems.append(f'{os.path.basename(f)}: {n} frames != {expect_frames}')
            if not (np.isfinite(m.rotations).all() and np.isfinite(m.positions).all()):
                problems.append(f'{os.path.basename(f)}: non-finite values')
        except Exception as e:  # noqa: BLE001 - report, don't crash the training run
            problems.append(f'{os.path.basename(f)}: {type(e).__name__}: {e}')
    return {'n_files': len(files), 'ok': len(files) > 0 and not problems, 'problems': problems}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_dir')
    parser.add_argument('--frames', type=int, default=55)
    args = parser.parse_args()
    report = check_bvh_dir(args.sample_dir, args.frames)
    print(json.dumps(report, indent=2))
    summary = os.path.join(args.sample_dir, 'summary.json')
    if os.path.exists(summary):
        print(open(summary).read())
    sys.exit(0 if report['ok'] else 1)
