"""Aggregate rollout and reference descriptors into the main table (paper Table "main").

    python scripts/eval_tables.py --runs ours=save/eval/released [other=save/eval/<run> ...] \\
        --ref save/eval/reference -o save/eval/tab_main.csv

Each `name=dir` expects `dir/<test_set>/descriptors.csv` for any of the test sets own / seen / withheld / heldout
(written by `python -m evaluation.run_metrics rollouts`). Every cell is the mean over the set's rollouts (the CSV
keeps the std next to it as `<column>_std`); a `provenance` column names the archive directory.

Rows:
    data  the reference half of the held-out takes retargeted to the other performers' bodies (MotionPersona-X):
          the physical columns of the data row. Its `fpd_*` columns are the protocol floor: each own-body query-half
          take against the reference half of the same (performer, style), as a rollout is scored.
    data_grid
          the reference half retargeted to the sweep and held-out grid bodies (the bodies of the cross-body sets)
    data_captured
          the reference half on the performer's own body, captured motion with its labels (diagnostic: captured feet
          pitch the sole corners into the floor, so its penetration is not comparable to a retargeted or generated body)
    <run> one row per test set; spread kept (`spread_dyn`, `spread_amp`) on the own-body set only.
Jerk is population-sensitive: the three data rows differ in it (0.56 / 0.51 / 0.54); compare a run's row with the
data row of the same kind of bodies (data_grid for the cross-body sets).
"""
import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)   # nanmean of an all-nan column -> nan, by design

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation import metrics as M  # noqa: E402
from evaluation import run_metrics as RM  # noqa: E402
from evaluation.cases import STYLES  # noqa: E402

SETS = ('own', 'seen', 'withheld', 'heldout')
QUALITY = ('fpd_pose', 'fpd_contact', 'toe_jerk_med', 'skate', 'iou', 'pen', 'seam_cm', 'track_cm', 'track_leg', 'speed_err_leg',
           'facing_err_deg', 'contact_ratio', 'h_contact_cm', 'toe_jerk', 'joint_jerk_med')
DATA_COLS = ('toe_jerk_med', 'skate', 'iou', 'pen', 'contact_ratio', 'h_contact_cm', 'toe_jerk', 'joint_jerk_med')


def read(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        if r.get('status', 'ok') != 'ok':
            continue
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v) if v not in ('', None) else np.nan
            except ValueError:
                d[k] = v
        out.append(d)
    return out


def mean_std(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std()), len(v)) if len(v) else (np.nan, np.nan, 0)


def own_body_of(bodies_csv):
    return {r['name']: int(r['id']) for r in csv.DictReader(open(bodies_csv)) if r['kind'] == 'real'}


def reference_cells(ref_rows):
    """Reference-half rows per (persona, body, style)."""
    ref = defaultdict(list)
    for r in ref_rows:
        if r['half'] == 'reference':
            ref[(r['subject'], int(r['body']), r['style'])].append(r)
    return ref


def spread(gen_rows, ref, group):
    """Spread kept: per descriptor, std over personas of the seed-mean generated value / std over personas of the
    mean of their reference-half takes on the same body, within a style, averaged over styles then over the group."""
    vals = []
    for k in group:
        per_style = []
        for s in STYLES:
            gen = defaultdict(list)
            for g in gen_rows:
                if g['style'] == s:
                    gen[(g['persona'], int(g['body']))].append(g[k])
            rf = {pb: float(np.nanmean([r[k] for r in ref[(*pb, s)]])) for pb in gen if (*pb, s) in ref}
            gm = {pb: float(np.nanmean(v)) for pb, v in gen.items() if np.isfinite(np.nanmean(v))}
            sk = M.spread_kept(gm, {pb: v for pb, v in rf.items() if np.isfinite(v)})
            if np.isfinite(sk):
                per_style.append(sk)
        if per_style:
            vals.append(float(np.mean(per_style)))
    return float(np.mean(vals)) if vals else np.nan


def fpd_floor(ref_dir, manifest, own_of):
    """The data's FPD: each own-body query-half take against the merged reference half of its (performer, style)."""
    by_cell = defaultdict(list)
    for r in manifest:
        if r['half'] == 'reference':
            by_cell[(r['subject'], int(r['body']), r['style'])].append(r)
    out = {f'fpd_{s}': [] for s in RM.SPACES}
    for r in manifest:
        if r['half'] != 'query' or own_of.get(r['subject']) != int(r['body']):
            continue
        cell = by_cell.get((r['subject'], int(r['body']), r['style']), [])
        for s in RM.SPACES:
            out[f'fpd_{s}'].append(RM.frechet_vs(RM.stats_gaussian(ref_dir, [r], s, min_frames=3),
                                                 RM.stats_gaussian(ref_dir, cell, s), ref_dir, s))
    return out


def data_row(method, test_set, provenance, rows):
    row = dict(method=method, test_set=test_set, provenance=provenance, n=len(rows))
    for k in DATA_COLS:
        m, s, _ = mean_std([r.get(k, np.nan) for r in rows]); row[k] = m; row[k + '_std'] = s
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='name=archive_dir (holding <test_set>/descriptors.csv)')
    ap.add_argument('--ref', default=str(RM.DEFAULT_REF))
    ap.add_argument('--bodies', default=str(REPO / 'data/bodies/bodies.csv'))
    ap.add_argument('-o', '--out', default=str(REPO / 'save/eval/tab_main.csv'))
    a = ap.parse_args()

    ref_dir = Path(a.ref)
    ref_rows = read(ref_dir / 'descriptors.csv'); ref = reference_cells(ref_rows)
    own_of = own_body_of(a.bodies)
    out = []
    # data: the reference half retargeted to the other performers' bodies + the FPD floor of the own-body halves
    x_ref = [r for r in ref_rows if int(r['body']) in set(own_of.values()) and own_of.get(r['subject']) != int(r['body']) and r['half'] == 'reference']
    row = data_row('data', 'real_bodies', str(ref_dir) + ' (retargeted to the other performers)', x_ref)
    for k, v in fpd_floor(ref_dir, list(csv.DictReader(open(ref_dir / 'manifest.csv'))), own_of).items():
        m, s, _ = mean_std(v); row[k] = m; row[k + '_std'] = s
    row['spread_dyn'] = row['spread_amp'] = 1.0
    out.append(row)
    grid_ref = [r for r in ref_rows if int(r['body']) not in set(own_of.values()) and r['half'] == 'reference']
    out.append(data_row('data_grid', 'grid_bodies', str(ref_dir) + ' (retargeted to the sweep / held-out bodies)', grid_ref))
    own_ref = [r for r in ref_rows if own_of.get(r['subject']) == int(r['body']) and r['half'] == 'reference']
    out.append(data_row('data_captured', 'own', str(ref_dir), own_ref))
    for spec in a.runs:
        name, base = spec.split('=', 1)
        for ts in SETS:
            p = Path(base) / ts / 'descriptors.csv'
            if not p.exists():
                continue
            rows = read(p)
            row = dict(method=name, test_set=ts, provenance=str(Path(base) / ts), n=len(rows))
            for k in QUALITY:
                m, s, _ = mean_std([r.get(k, np.nan) for r in rows]); row[k] = m; row[k + '_std'] = s
            if ts == 'own':
                row['spread_dyn'] = spread(rows, ref, M.SPREAD_DYN); row['spread_amp'] = spread(rows, ref, M.SPREAD_AMP)
            out.append(row)
    write(a.out, out)
    show = ('fpd_pose', 'fpd_contact', 'toe_jerk_med', 'skate', 'iou', 'pen', 'seam_cm', 'track_cm', 'spread_dyn', 'spread_amp')
    print(f'{"method":14s} {"test_set":12s} ' + ' '.join(f'{k[:11]:>11s}' for k in show))
    for r in out:
        print(f'{r["method"]:14s} {r["test_set"]:12s} ' + ' '.join(f'{r.get(k, np.nan):11.4f}' for k in show))


def write(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in r.items()})
    print(f'-> {path} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
