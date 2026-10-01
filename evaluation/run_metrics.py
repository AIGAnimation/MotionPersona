"""Batch metrics over a rollout archive or the reference set.

    python -m evaluation.run_metrics refs  --ref save/eval/reference [--workers 8]
        -> reference/descriptors.csv (one row per reference take x body: quality metrics with the captured labels as
           the contact flag + spread descriptors)
           reference/fpd_stats_v<FPD_VERSION>/<body>/<stem>.npz (sufficient statistics of the pose and contact
           features) + _scale.npz (per-dimension std of each space over the own-body reference-half takes: the unit of
           the standardised contact FPD)
    python -m evaluation.run_metrics rollouts --dir save/eval/<run>/<test_set> --ref save/eval/reference [--workers 8]
        -> <dir>/descriptors.csv (one row per rollout: case fields, quality, tracking, seam, fpd_pose + fpd_contact vs
           the reference half of the same (persona, body, style), spread descriptors)

FPD reference = every REFERENCE-half take of that persona in that style on that body, merged from per-take sufficient
statistics (n, sum, sum of outer products) -> exact Gaussian of all frames. The statistics directory is versioned
(metrics.FPD_VERSION): a file of another version is refused, so a rollout is never scored against stale statistics.
"""
import os
# One thread per worker process: without this every Pool worker spins up one OMP/torch thread per core.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation import metrics as M  # noqa: E402

FEET = REPO / 'data/eval/feet_local.npz'
DEFAULT_REF = REPO / 'save/eval/reference'
STATS_DIR = f'fpd_stats_v{M.FPD_VERSION}'
SCALE_FILE = '_scale.npz'
SPACES = ('pose', 'contact')
STANDARDISED = ('contact',)                                          # spaces measured in units of the reference std
_feet = None
_scale = {}


def feet_of(body):
    global _feet
    if _feet is None:
        z = np.load(FEET); _feet = dict(zip(z['body'].tolist(), z['feet']))
    return _feet[int(body)]


# ------------------------------------------------------------------------------------------- reference set
def _ref_one(args):
    ref_dir, rel = args
    p = Path(ref_dir) / rel
    z = np.load(p)
    head = dict(path=rel, stem=p.stem, body=int(z['body']), subject=str(z['subject']), style=str(z['style']), half=str(z['half']))
    try:
        out, feats = M.reference_metrics(z, feet_local=feet_of(int(z['body'])))
        st = Path(ref_dir) / STATS_DIR / rel
        st.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(version=M.FPD_VERSION)
        for k, feat in feats.items():
            n, s1, s2 = M.sufficient_stats(feat)
            payload[f'{k}_n'] = n; payload[f'{k}_s1'] = s1; payload[f'{k}_s2'] = s2
        np.savez(st, **payload)
        out.update(head, status='ok')
    except Exception as e:                                                   # keep going; the row says why
        out = dict(head, status=f'error: {type(e).__name__}: {e}')
    return out


def cmd_refs(a):
    ref = Path(a.ref)
    rows = list(csv.DictReader(open(ref / 'manifest.csv')))
    todo = [(str(ref), r['path']) for r in rows]
    if a.limit:
        todo = todo[:a.limit]
    with Pool(a.workers, initializer=_worker_init) as pool:
        outs = list(pool.imap(_ref_one, todo, chunksize=8))
    write_rows(ref / 'descriptors.csv', outs)
    print(f'{len(outs)} reference rows -> {ref / "descriptors.csv"} ({sum(o["status"] != "ok" for o in outs)} errors)')
    if not a.limit:
        write_scale(ref, rows)


def write_scale(ref_dir, manifest):
    """reference/<STATS_DIR>/_scale.npz: per-dimension std of every space over the own-body reference-half takes (the
    unit of the standardised fpd_contact; the pose space is used in leg units and its scale is only recorded)."""
    ref_dir = Path(ref_dir)
    rows = [r for r in manifest if r['half'] == 'reference' and r.get('role', 'own') == 'own']
    payload = dict(version=M.FPD_VERSION, n_takes=len(rows))
    for space in SPACES:
        parts = []
        for r in rows:
            p = ref_dir / STATS_DIR / r['path']
            if p.exists():
                z = load_stats(p); parts.append((z[f'{space}_n'], z[f'{space}_s1'], z[f'{space}_s2']))
        n, mu, cov = M.merge_stats(parts)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        payload[f'{space}_mu'] = mu; payload[f'{space}_std'] = np.where(std > 1e-9, std, 1.0); payload[f'{space}_n'] = n
    np.savez(ref_dir / STATS_DIR / SCALE_FILE, **payload)
    print(f'scale of {len(rows)} own reference-half takes -> {ref_dir / STATS_DIR / SCALE_FILE}')


def load_stats(path):
    z = np.load(path)
    if 'version' not in z or int(z['version']) != M.FPD_VERSION:
        raise RuntimeError(f'{path}: FPD statistics version {z["version"] if "version" in z else "none"} != {M.FPD_VERSION}; rerun `run_metrics refs`')
    return z


def scale_of(ref_dir, space):
    """Per-dimension std of `space` over the own reference-half takes (None if the refs run has not written it)."""
    key = (str(ref_dir), space)
    if key not in _scale:
        p = Path(ref_dir) / STATS_DIR / SCALE_FILE
        _scale[key] = load_stats(p)[f'{space}_std'] if p.exists() else None
    return _scale[key]


def stats_gaussian(ref_dir, rows, space, min_frames=10):
    """Merged Gaussian (mu, cov) of the given reference rows in `space`, or None when they hold too few frames."""
    parts = []
    for r in rows:
        p = Path(ref_dir) / STATS_DIR / r['path']
        if p.exists():
            z = load_stats(p); parts.append((z[f'{space}_n'], z[f'{space}_s1'], z[f'{space}_s2']))
    if not parts or sum(int(p[0]) for p in parts) < min_frames:
        return None
    n, mu, cov = M.merge_stats(parts)
    return mu, cov


def fpd_reference(ref_dir, persona, body, style, manifest=None, space='pose'):
    """Merged Gaussian of the reference-half takes of (persona, style) on `body` in `space` -> (mu, cov) or None."""
    if manifest is None:
        manifest = list(csv.DictReader(open(Path(ref_dir) / 'manifest.csv')))
    rows = [r for r in manifest if r['subject'] == persona and r['style'] == style and int(r['body']) == int(body) and r['half'] == 'reference']
    return stats_gaussian(ref_dir, rows, space)


def frechet_vs(gen, ref, ref_dir, space):
    """Frechet distance of two Gaussians (mu, cov), in units of the reference std for the standardised spaces."""
    if gen is None or ref is None:
        return np.nan
    if space in STANDARDISED:
        sc = scale_of(ref_dir, space)
        if sc is None:
            return np.nan
        gen, ref = M.standardise(*gen, sc), M.standardise(*ref, sc)
    return M.frechet(*gen, *ref)


def fpd_columns(feats, ref_dir, persona, body, style, manifest):
    """`fpd_pose` (leg units) and `fpd_contact` (standardised) against the merged reference Gaussians of (persona, body, style)."""
    out = {}
    for space in SPACES:
        g = M.gaussian_fit(feats[space]) if space in feats and len(feats[space]) >= 3 else None
        out[f'fpd_{space}'] = frechet_vs(g, fpd_reference(ref_dir, persona, body, style, manifest, space=space), ref_dir, space)
    return out


# ------------------------------------------------------------------------------------------- rollouts
def _roll_one(args):
    d, stem, ref_dir, manifest = args
    d = Path(d)
    try:
        r = M.Rollout.load(d / f'{stem}.motion.npz', d / f'{stem}.traj.npz')
        meta = json.load(open(d / f'{stem}.meta.json'))
        case = meta['case']
        out, feats = M.rollout_metrics(r, feet_local=feet_of(r.body))
        if ref_dir:
            out.update(fpd_columns(feats, ref_dir, case['persona'], case['body'], case['style'], manifest))
        out.update(stem=stem, persona=case['persona'], body=int(case['body']), style=case['style'], seed=int(case['seed']),
                   run=meta.get('run_name'), test_set=meta.get('test_set'), leg=r.leg, ms_per_block=meta.get('ms_per_block_mean'), status='ok')
    except Exception as e:
        out = dict(stem=stem, status=f'error: {type(e).__name__}: {e}')
    return out


def cmd_rollouts(a):
    d = Path(a.dir)
    stems = sorted(p.name[:-len('.motion.npz')] for p in d.glob('*.motion.npz'))
    if a.limit:
        stems = stems[:a.limit]
    ref = a.ref if a.ref and (Path(a.ref) / 'manifest.csv').exists() else None
    if ref is None:
        print(f'no reference set at {a.ref}: fpd columns skipped (scripts/make_eval_refs.py + `run_metrics refs` build it)')
    manifest = list(csv.DictReader(open(Path(ref) / 'manifest.csv'))) if ref else None
    with Pool(a.workers, initializer=_worker_init) as pool:
        outs = list(pool.imap(_roll_one, [(str(d), s, ref, manifest) for s in stems], chunksize=4))
    out = Path(a.out) if a.out else d / 'descriptors.csv'
    write_rows(out, outs)
    ok = [o for o in outs if o['status'] == 'ok']
    print(f'{len(outs)} rollouts -> {out} ({len(outs) - len(ok)} errors)')
    if ok:
        for k in ('fpd_pose', 'fpd_contact', 'toe_jerk_med', 'skate', 'iou', 'pen', 'seam_cm', 'track_cm', 'facing_err_deg', 'contact_ratio'):
            v = np.array([o.get(k, np.nan) for o in ok], float)
            if np.isfinite(v).any():
                print(f'  {k:16s} mean {np.nanmean(v):.3f}  median {np.nanmedian(v):.3f}')


def _worker_init():
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(10)
    except OSError:
        pass


def write_rows(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    front = [k for k in ('stem', 'run', 'test_set', 'persona', 'subject', 'body', 'style', 'seed', 'half', 'status') if k in keys]
    keys = front + [k for k in keys if k not in front]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('refs'); r.add_argument('--ref', default=str(DEFAULT_REF)); r.add_argument('--workers', type=int, default=8); r.add_argument('--limit', type=int, default=0); r.set_defaults(fn=cmd_refs)
    g = sub.add_parser('rollouts'); g.add_argument('--dir', required=True); g.add_argument('--ref', default=str(DEFAULT_REF))
    g.add_argument('--out', default=None, help='output CSV (default <dir>/descriptors.csv)')
    g.add_argument('--workers', type=int, default=8); g.add_argument('--limit', type=int, default=0); g.set_defaults(fn=cmd_rollouts)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
