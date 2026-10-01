"""The evaluation reference set: the held-out takes, straight from the retargeted flat dataset.

    python scripts/make_eval_refs.py --data <flat dataset> [--split data/eval/split_v2] [-o save/eval/reference] [--no-bvh]

The held-out takes (data/eval/split_v2/heldout_takes.csv; split in a reference and a query half) never entered
training. Rows written, each the take retargeted to one body by the production retargeter:
    own      both halves on the performer's own body       (FPD references of the own-body set + the data FPD floor)
    sweep    reference half on the 16 sweep bodies          (FPD references of the seen / withheld sets)
    heldout  reference half on the 15 held-out bodies       (FPD references of the held-out set)
    real     reference half on the other performers' bodies (the data row of the main table)
-> <out>/<body:03d>/<stem>.npz (+ .bvh unless --no-bvh) and <out>/manifest.csv. Score them with
`python -m evaluation.run_metrics refs --ref <out>`.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.archive import write_bvh  # noqa: E402
from evaluation.route import file_sha256  # noqa: E402


def read_split(path):
    return [r for r in csv.DictReader(l for l in open(path) if not l.startswith('#'))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='flat dataset directory (manifest.csv, frames/, motions/)')
    ap.add_argument('--split', default=str(REPO / 'data/eval/split_v2'))
    ap.add_argument('--bodies-csv', default=str(REPO / 'data/bodies/bodies.csv'))
    ap.add_argument('-o', '--out', default=str(REPO / 'save/eval/reference'))
    ap.add_argument('--no-bvh', action='store_true'); ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    d = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sp = Path(a.split)
    takes = {r['stem']: r for r in read_split(sp / 'heldout_takes.csv')}
    sweep = [int(r['body']) for r in read_split(sp / 'sweep_bodies.csv')]
    held = [int(r['body']) for r in read_split(sp / 'heldout_bodies.csv')]
    bodies = {int(r['id']): r for r in csv.DictReader(open(a.bodies_csv))}
    own_of = {r['name']: b for b, r in bodies.items() if r['kind'] == 'real'}
    performers = {t['subject'] for t in takes.values()}
    real = {own_of[p] for p in performers if p in own_of}              # the performers' own bodies

    def role(r):
        b = int(r['body'])
        if b == own_of.get(r['subject']):
            return 'own'
        return 'heldout' if b in held else ('sweep' if b in sweep else ('real' if b in real else None))

    rows = []
    for r in csv.DictReader(open(d / 'manifest.csv')):
        t = takes.get(r['stem'])
        if t is None:
            continue
        ro = role(r)
        if ro == 'own' or (ro is not None and t['half'] == 'reference'):     # query halves only on the own body
            rows.append(dict(r, role=ro))
    by_role = {ro: sum(r['role'] == ro for r in rows) for ro in ('own', 'sweep', 'heldout', 'real')}
    print(f'{len(rows)} rows of {len(takes)} held-out takes: ' + ', '.join(f'{n} {ro}' for ro, n in by_role.items()))
    off = np.load(d / 'motions/frame_offsets.npy')
    rot = np.load(d / 'frames/rotations.npy', mmap_mode='r'); rp = np.load(d / 'frames/root_pos.npy', mmap_mode='r')
    fc = np.load(d / 'frames/foot_contact.npy', mmap_mode='r'); so = np.load(d / 'motions/skel_offset.npy', mmap_mode='r')
    sf = np.load(d / 'motions/shape_feat.npy', mmap_mode='r')
    if a.no_bvh:
        T_pose = None
    else:
        from data.loco_dataset import read_meta
        T_pose = read_meta(d if (d / 'meta.pkl').exists() else REPO / 'data/skeleton')['T_pose']
    man = []
    for k, r in enumerate(sorted(rows, key=lambda r: (int(r['body']), r['stem']))):
        i = int(r['idx']); s, e = int(off[i]), int(off[i + 1]); b = int(r['body']); t = takes[r['stem']]
        bd = out / f'{b:03d}'; bd.mkdir(exist_ok=True)
        p = bd / f'{r["stem"]}.npz'
        if not p.exists() or a.force:
            np.savez(p, rotations=np.array(rot[s:e]), root_pos=np.array(rp[s:e]), foot_contact=np.array(fc[s:e]),
                     skel_offset=np.array(so[i]), betas=np.array(sf[i]), body=b, source_idx=int(r['src']), motion_idx=i,
                     subject=r['subject'], style=r['style'], half=t['half'], kind=r['kind'], status=r['status'])
            if T_pose is not None:
                write_bvh(bd / f'{r["stem"]}.bvh', T_pose, np.array(so[i]), np.array(rot[s:e]), np.array(rp[s:e]))
        man.append(dict(path=f'{b:03d}/{r["stem"]}.npz', stem=r['stem'], subject=r['subject'], style=r['style'], half=t['half'],
                        body=b, body_name=r['body_name'], kind=r['kind'], role=r['role'], n_frames=e - s, status=r['status'],
                        src=int(r['src']), idx=i))
        if k % 2000 == 0:
            print(f'  {k}/{len(rows)}')
    with open(out / 'manifest.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(man[0])); w.writeheader(); w.writerows(man)
    json.dump({'data': str(d), 'split': str(sp), 'split_files': {n: file_sha256(sp / f'{n}.csv') for n in ('heldout_takes', 'sweep_bodies', 'heldout_bodies')},
               'n_rows': len(man), 'bvh': not a.no_bvh}, open(out / 'index.json', 'w'), indent=1)
    print(f'{len(man)} reference rows -> {out}')


if __name__ == '__main__':
    main()
