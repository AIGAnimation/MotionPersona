"""The rollout archive: save/eval/<run>/<test_set>/<persona>__<body>__<style>__s<seed>.*

    .bvh          1,800 frames, 30 fps, cm, Euler XYZ, root CHANNELS 6, the body's skeleton, y = 0 = sole plane;
                  the raw model output (no foot lock, no cross-fade). History is NOT included (data/eval/canonical/).
    .motion.npz   float32 quats (n,J,4) / root (n,3) m / contact (n,2) / skel_offset / betas -- lossless, for metrics
    .contacts.npz the model's contact flags
    .traj.npz     the body-scaled command sequence + the actual root path / speed / facing, block timing
    .meta.json    provenance: checkpoints (sha256), git, config, route / canonical / caps hashes, seed, head, settings
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from evaluation.engine import CaseResult, facing_of
from evaluation.route import file_sha256

FPS = 30


def git_state(repo):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip())
        return {'commit': commit, 'dirty': dirty}
    except Exception:                                                                   # not a checkout
        return {'commit': None, 'dirty': None}


def write_bvh(path, T_pose, offsets, quats, root):
    tpl = T_pose.copy()                                   # export divides end_offsets in place: never reuse a template
    tpl.offsets = np.asarray(offsets, np.float64)
    tpl.rotations = np.asarray(quats, np.float64)
    pos = np.zeros((len(root), tpl.offsets.shape[0], 3))
    pos[:, 0] = root
    tpl.positions = pos
    tpl.export(str(path), save_ori_scal=True)


class Archive:
    def __init__(self, out_dir, bundle, *, run_name, test_set, route_sha, canonical_index_sha=None, caps_sha=None,
                 persona_rows_sha=None, split_shas=None, settings=None, repo=None):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.b = bundle
        self.run_name, self.test_set = run_name, test_set
        self.static = {
            'run_name': run_name, 'test_set': test_set,
            'ckpts': {'model': {'path': bundle.ckpt, 'sha256': bundle.sha256.get('ckpt')},
                      'codec': {'sha256': bundle.sha256.get('codec')} if 'codec' in bundle.sha256 else None},
            'head': bundle.kind, 'ema': bundle.ema, 'K': bundle.K, 'F': bundle.F, 'J': bundle.J,
            'root_basis': bundle.root_basis,
            'config_sha256': hashlib.sha256(OmegaConf.to_yaml(bundle.cfg).encode()).hexdigest(),
            'git': git_state(repo or Path(__file__).resolve().parents[1]),
            'route_sha256': route_sha, 'canonical_index_sha256': canonical_index_sha, 'speed_caps_sha256': caps_sha,
            'persona_rows_sha256': persona_rows_sha, 'split_files': split_shas or {},
            'settings': settings or {},
            'units': 'BVH: cm, 30 fps, Euler XYZ, root CHANNELS 6, y=0 = sole plane; motion.npz: metres, wxyz quats',
        }
        self.rows = []

    def write(self, res: CaseResult):
        stem = res.case.key
        bvh = self.dir / f'{stem}.bvh'
        write_bvh(bvh, self.b.T_pose, res.skel_offset, res.quats, res.root)
        np.savez(self.dir / f'{stem}.motion.npz', quats=res.quats.astype(np.float32), root=res.root.astype(np.float32),
                 contact=res.contact.astype(np.float32), skel_offset=res.skel_offset.astype(np.float32), betas=res.betas.astype(np.float32),
                 leg=res.leg, body=res.case.body)
        np.savez(self.dir / f'{stem}.contacts.npz', contact=res.contact.astype(np.float32),
                 left=res.contact[:, 0] > 0.5, right=res.contact[:, 1] > 0.5)
        n = len(res.root)
        root_xz = res.root[:, [0, 2]]
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(root_xz, axis=0), axis=-1) * FPS])
        frame_of_block = np.arange(len(res.block_ms)) * (n // len(res.block_ms))
        np.savez(self.dir / f'{stem}.traj.npz', cmd_xz_m=res.cmd_xz_m.astype(np.float32), cmd_speed_mps=res.cmd_speed_mps.astype(np.float32),
                 cmd_heading_rad=res.cmd_heading.astype(np.float32), cmd_facing_rad=res.cmd_facing.astype(np.float32),
                 root_xz_m=root_xz.astype(np.float32), root_y_m=res.root[:, 1].astype(np.float32), root_speed_mps=speed.astype(np.float32),
                 root_facing_rad=facing_of(res.quats, res.skel_offset).astype(np.float32), leg=res.leg, speed_scale=res.speed_scale,
                 v_cap_leg=-1.0 if res.v_cap_leg is None else res.v_cap_leg, block_ms=res.block_ms, frame_of_block=frame_of_block)
        meta = dict(self.static, case={'persona': res.case.persona, 'body': res.case.body, 'style': res.case.style, 'seed': res.case.seed},
                    leg=res.leg, speed_scale=res.speed_scale, v_cap_leg=res.v_cap_leg, canonical=res.canonical, frames=n,
                    n_blocks=int(len(res.block_ms)), hop=n // len(res.block_ms), ms_per_block_mean=float(res.block_ms.mean()),
                    batch_size=res.batch_size, batch_index=res.batch_index, bvh_sha256=file_sha256(bvh),
                    written=time.strftime('%Y-%m-%d %H:%M:%S'), **res.extra)
        json.dump(meta, open(self.dir / f'{stem}.meta.json', 'w'), indent=1)
        row = dict(path=bvh.name, sha256=meta['bvh_sha256'], persona=res.case.persona, body=res.case.body, style=res.case.style,
                   seed=res.case.seed, frames=n, speed_scale=f'{res.speed_scale:.6f}', ms_per_block=f'{res.block_ms.mean():.2f}',
                   batch_index=res.batch_index, status='ok')
        self.rows.append(row)
        return row

    def done(self, stem):
        return (self.dir / f'{stem}.meta.json').exists()

    def flush_manifest(self):
        import csv
        p = self.dir / 'manifest.csv'
        old = []
        if p.exists():
            old = [r for r in csv.DictReader(open(p)) if r['path'] not in {r2['path'] for r2 in self.rows}]
        rows = sorted(old + self.rows, key=lambda r: r['path'])
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['path', 'sha256', 'persona', 'body', 'style', 'seed', 'frames', 'speed_scale', 'ms_per_block', 'batch_index', 'status'])
            w.writeheader(); w.writerows(rows)
        return p
