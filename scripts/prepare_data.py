"""MotionPersonaX (Hugging Face dataset) -> the ``flat_v1`` training dataset (layout: data/flat_writer.py).

    python scripts/prepare_data.py --src <MotionPersonaX dir>                                   # all 128 bodies
    python scripts/prepare_data.py --src <MotionPersonaX dir> --bodies p02 b27 grid_h05_g03 --out data/mpx_subset

Reads ``smpl_skeleton/smpl/<body>.tar.zst`` (one SMPL-X npz per clip, streamed, never unpacked to disk) and, when
present, ``manifest.csv`` (the ``status`` column).  The SMPL-X model files are not needed: the pelvis rest position
and the 23-joint skeleton are linear in the betas (``data/bodies/joint_regressor.npz``).

Per clip:
  * ``rotations``: local joint rotations of the 23 body joints (SMPL-X joints 0-21 + jaw) as wxyz quaternions
    (w >= 0; every model input is a rotation matrix, so the sign is immaterial), root orientation in a Y-up frame;
  * ``root_pos``: pelvis world position in metres, Y-up (y = 0 is the sole plane of the body);
  * ``traj_pose``: facing of the hips, gaussian-smoothed with sigma 3 and 6 frames;
  * ``foot_contact``: the labels of the source capture (``data/motionpersonax_sources.npz``; a retargeted clip is
    frame-aligned with its source and keeps its labels, as in training);
  * ``skel_offset`` / ``shape_feat``: skeleton of the target body (sole basis, ``skeleton``) / its betas.

Motion order is the one of the paper's dataset: slot 0 = every source clip on its performer's own body, then slot
``b + 1`` = every source clip on body id ``b`` (own body skipped), source clips in the table order.  A ``--bodies``
subset keeps that order.  ``manifest.csv`` stems are the MotionPersonaX clip names (what ``data/eval/split_v2``
uses for held-out takes); ``meta.pkl`` carries the skeleton template and the root normalisation statistics of the
full dataset (``data/skeleton/meta.pkl``, the ones the released checkpoints use) unless ``--recompute-stats``.
CLIP text features are not written: the released codec and prior do not read them (the loader serves zeros).

Frames are used as stored (30 fps nominal; frame 0 is the reference pose of the capture).  The 120 source clips
whose ``mocap_frame_rate`` was corrected in the release (60 or 15 fps, 15,360 retargeted clips) are not resampled,
exactly as in the paper.

Resumable: finished bodies are recorded under ``<out>/.prepare``; rerun the same command after an interruption.
Full dataset: 328,960 motions, 453.5 M frames, about 190 GB written.
"""
import argparse
import csv
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data.bodies import BODIES_DIR, REPO_NAMES, betas_to_offsets, load_regressor   # noqa: E402
from data.flat_writer import LAYOUT                                           # noqa: E402
from utils.bvh_motion import Motion                                           # noqa: E402
from utils.motion_processing import extract_forward_hips                      # noqa: E402

SOURCES = REPO / 'data/motionpersonax_sources.npz'
BODIES = REPO / 'data/bodies/bodies.npz'
TEMPLATE = REPO / 'data/skeleton/meta.pkl'
SMPLX_BODY = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle',
              'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head',
              'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw']
FLAT_FROM_SMPLX = np.array([SMPLX_BODY.index(n) for n in REPO_NAMES])
HIPS = [REPO_NAMES.index(n) for n in ('pelvis', 'left_hip', 'right_hip')]
R_YUP_TO_ZUP = R.from_rotvec([np.pi / 2, 0, 0])            # the release's frame change (SMPL-X npz are Z-up)
TRAJ_SIGMAS = (3, 6)
STYLE_CATEGORY = {'angry': 'emotional', 'depressed': 'emotional', 'fear': 'emotional', 'happy': 'emotional',
                  'neutral': 'emotional', 'bigstep': 'performing', 'drunk': 'performing', 'swimming': 'performing',
                  'twofootjump': 'performing'}
BYTES_PER_FRAME = 23 * 4 * 4 + 3 * 4 + 2 * 4 + 2 * 4 * 4
FRAME_SPECS = {'rotations': (23, 4), 'root_pos': (3,), 'foot_contact': (2,), 'traj_pose': (len(TRAJ_SIGMAS), 4)}


# --------------------------------------------------------------------------- conversion

def smplx_to_flat(d, j0):
    """SMPL-X npz dict (Z-up) -> rotations (T,23,4) wxyz float32, root_pos (T,3) float32 (pelvis world, Y-up)."""
    up = R_YUP_TO_ZUP.inv()
    T = len(d['trans'])
    root = up * R.from_rotvec(np.asarray(d['root_orient'], np.float64))
    aa = np.concatenate([root.as_rotvec()[:, None],
                         np.asarray(d['pose_body'], np.float64).reshape(T, 21, 3),
                         np.asarray(d['pose_jaw'], np.float64)[:, None]], 1)              # (T,23,3) SMPL-X order
    q = R.from_rotvec(aa.reshape(-1, 3)).as_quat().reshape(T, 23, 4)[:, FLAT_FROM_SMPLX][..., [3, 0, 1, 2]]
    q *= np.where(q[..., :1] < 0, -1.0, 1.0)
    pos = up.apply(np.asarray(d['trans'], np.float64) + j0)
    return q.astype(np.float32), pos.astype(np.float32)


def skeleton(beta, body, reg):
    """(23,3) skel_offset of a body: bones from the betas, root row = pelvis above the sole plane with the
    per-body sole drop measured on the mesh (data/bodies/sole_basis.npz, the values the training data use)."""
    off = betas_to_offsets(beta[None], reg, root_basis='joint')[0]
    sb = np.load(BODIES_DIR / 'sole_basis.npz')
    off[0, 1] += sb['drop_per_body'][list(sb['names']).index(body)]
    return off


def traj_pose(q, root, offsets):
    """(T,A,4) wxyz facing quaternions: hip forward smoothed with each of TRAJ_SIGMAS (the training-data feature)."""
    T = len(q)
    pos = np.zeros((T, 3, 3), np.float32)
    pos[:, 0] = root
    m = Motion(q[:, HIPS].astype(np.float32), pos, np.asarray(offsets, np.float32)[HIPS], np.array([-1, 0, 0]),
               ['pelvis', 'left_hip', 'right_hip'], 1 / 30)
    _, forwards = extract_forward_hips(m, np.arange(T), 'left_hip', 'right_hip', return_forward=True)
    out = []
    v0 = np.array([[0, 0, 1]]).repeat(T, axis=0)
    for sigma in TRAJ_SIGMAS:
        f = gaussian_filter1d(forwards, sigma, axis=0, mode='nearest')
        f = f / np.linalg.norm(f, axis=-1, keepdims=True)
        a = np.cross(v0, f)
        w = np.sqrt((v0 ** 2).sum(axis=-1) * (f ** 2).sum(axis=-1)) + (v0 * f).sum(axis=-1)
        between = np.concatenate([w[..., None], a], axis=-1)
        out.append(R.from_quat(between[..., [1, 2, 3, 0]]).as_quat()[..., [3, 0, 1, 2]].astype(np.float32))
    return np.stack(out, 1)


def open_archive(path):
    """Stream the members of a .tar.zst: python-zstandard if installed, else the zstd command line tool."""
    try:
        import zstandard
        fh = open(path, 'rb')
        stream = zstandard.ZstdDecompressor().stream_reader(fh, read_size=1 << 20)
        return tarfile.open(fileobj=stream, mode='r|'), lambda: (stream.close(), fh.close())
    except ImportError:
        if shutil.which('zstd') is None:
            raise SystemExit('reading .tar.zst needs `pip install zstandard` or the zstd command line tool')
        proc = subprocess.Popen(['zstd', '-dcq', str(path)], stdout=subprocess.PIPE)
        return tarfile.open(fileobj=proc.stdout, mode='r|'), lambda: (proc.stdout.close(), proc.wait())


# --------------------------------------------------------------------------- plan

def load_sources():
    z = np.load(SOURCES)
    n = z['n_frames'].astype(np.int64)
    contact = np.unpackbits(z['contact'], count=int(n.sum()) * 2).reshape(-1, 2).astype(np.float32)
    attrs = {p: dict(role=str(r), affiliation=str(a), dominance=str(d))
             for p, r, a, d in zip(z['persona'], z['role'], z['affiliation'], z['dominance'])}
    return dict(clip=[str(c) for c in z['clip']], src=z['src'].astype(np.int64), n_frames=n,
                c_off=np.concatenate([[0], np.cumsum(n)]), contact=contact, attrs=attrs)


def make_plan(src_dir, bodies_sel):
    """Rows (k = source index in the table, body id, slot) in dataset order + per-body betas."""
    S = load_sources()
    bd = np.load(BODIES)
    name_of = {int(i): str(n) for i, n in zip(bd['id'], bd['name'])}
    id_of = {n: i for i, n in name_of.items()}
    kind_of = {int(i): str(k) for i, k in zip(bd['id'], bd['kind'])}
    arch = Path(src_dir) / 'smpl_skeleton/smpl'
    names = bodies_sel or sorted(p.name[:-len('.tar.zst')] for p in arch.glob('*.tar.zst'))
    unknown = [b for b in names if b not in id_of]
    if unknown:
        raise SystemExit(f'unknown bodies {unknown}')
    missing = [b for b in names if not (arch / f'{b}.tar.zst').exists()]
    if missing:
        raise SystemExit(f'archives missing under {arch}: {missing}')
    sel = sorted(id_of[b] for b in names)
    own = [id_of[c.split('_')[0]] for c in S['clip']]
    rows = [(k, own[k], 0) for k in range(len(own)) if own[k] in sel]
    for b in sel:
        rows += [(k, b, b + 1) for k in range(len(own)) if own[k] != b]
    betas = {}
    for b in sel:
        sh = Path(src_dir) / f'smpl_skeleton/shapes/{name_of[b]}.npz'
        beta = np.load(sh)['betas'][:10] if sh.exists() else bd['betas'][list(bd['id']).index(b)]
        if not np.allclose(beta, bd['betas'][list(bd['id']).index(b)], atol=1e-6):
            raise SystemExit(f'{sh}: betas differ from data/bodies/bodies.npz')
        betas[b] = np.asarray(beta, np.float64)
    return S, rows, sel, betas, name_of, kind_of


# --------------------------------------------------------------------------- per body

def body_job(job):
    out, src_dir, b, body, beta, entries, c_off, done_path = job
    t0 = time.time()
    contact = _G['contact']
    reg = load_regressor()
    offsets = skeleton(beta, body, reg)
    j0 = reg['J0'][0] + reg['A'][0] @ beta                                 # SMPL-X pelvis rest position (model frame)
    mm = {k: np.load(Path(out) / 'frames' / f'{k}.npy', mmap_mode='r+') for k in FRAME_SPECS}
    want = {clip: (k, s) for clip, k, s in entries}                        # clip -> (source index, global frame start)
    seen, acc, acc2 = set(), np.zeros(3), np.zeros(3)
    tf, close = open_archive(Path(src_dir) / f'smpl_skeleton/smpl/{body}.tar.zst')
    for mem in tf:
        if not mem.isfile() or not mem.name.endswith('.npz'):
            continue
        clip = Path(mem.name).stem
        if clip not in want:
            continue
        d = np.load(io.BytesIO(tf.extractfile(mem).read()))
        k, s = want[clip]
        q, root = smplx_to_flat(d, j0)
        T = len(q)
        if T != c_off[k + 1] - c_off[k]:
            raise RuntimeError(f'{body}/{clip}: {T} frames, expected {c_off[k + 1] - c_off[k]}')
        if not np.allclose(d['betas'][:10], beta, atol=1e-6):
            raise RuntimeError(f'{body}/{clip}: betas differ from the body table')
        mm['rotations'][s:s + T] = q
        mm['root_pos'][s:s + T] = root
        mm['foot_contact'][s:s + T] = contact[c_off[k]:c_off[k + 1]]
        mm['traj_pose'][s:s + T] = traj_pose(q, root, offsets)
        r64 = root.astype(np.float64)
        acc += r64.sum(0)
        acc2 += (r64 ** 2).sum(0)
        seen.add(clip)
    close()
    if seen != set(want):
        raise RuntimeError(f'{body}: {len(set(want) - seen)} clips missing from the archive')
    for a in mm.values():
        a.flush()
    del mm
    json.dump(dict(body=body, clips=len(seen), sum=acc.tolist(), sum2=acc2.tolist(), seconds=round(time.time() - t0)),
              open(done_path, 'w'))
    return body, len(seen), time.time() - t0


_G = {}


def _init(contact):
    _G['contact'] = contact


# --------------------------------------------------------------------------- main

def write_tables(out, src_dir, S, rows, offsets_by_row, name_of, kind_of, betas):
    """motions/*.npy, meta.jsonl, manifest.csv."""
    out = Path(out)
    status = {}
    man_x = Path(src_dir) / 'manifest.csv'
    if man_x.exists():
        for r in csv.DictReader(open(man_x)):
            status[(r['body'], r['clip'])] = r['status']
    reg = load_regressor()
    skel = {b: skeleton(betas[b], name_of[b], reg) for b in betas}
    np.save(out / 'motions/frame_offsets.npy', offsets_by_row)
    np.save(out / 'motions/shape_feat.npy', np.stack([betas[b].astype(np.float32) for _, b, _ in rows]))
    np.save(out / 'motions/skel_offset.npy', np.stack([skel[b] for _, b, _ in rows]))
    with open(out / 'meta.jsonl', 'w') as fj, open(out / 'manifest.csv', 'w', newline='') as fm:
        w = csv.writer(fm)
        w.writerow(['idx', 'src', 'subject', 'stem', 'style', 'body', 'body_name', 'kind', 'slot', 'n_frames', 'status'])
        for i, (k, b, slot) in enumerate(rows):
            clip, body = S['clip'][k], name_of[b]
            subject, style = clip.split('_')[:2]
            kind = 'own' if slot == 0 else kind_of[b]
            n = int(S['n_frames'][k])
            label = dict(style=style, style_category=STYLE_CATEGORY[style], **S['attrs'][subject], target_body=b,
                         target_name=body, target_kind=kind, source_idx=int(S['src'][k]), slot=slot)
            fj.write(json.dumps(dict(idx=i, n_frames=n, frame_offset=int(offsets_by_row[i]), filepath=f'{body}/{clip}.npz',
                                     skel_name=body, mirror=False, text_raw=None, label=label)) + '\n')
            w.writerow([i, int(S['src'][k]), subject, clip, style, b, body, kind, slot, n, status.get((body, clip), '')])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--src', required=True, help='MotionPersonaX directory (smpl_skeleton/smpl/*.tar.zst)')
    ap.add_argument('--out', default='data/motionpersonax_flat')
    ap.add_argument('--bodies', nargs='*', help='target bodies to convert (default: every archive under --src)')
    ap.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument('--recompute-stats', action='store_true',
                    help='root_pos mean/std of the converted frames instead of the full-dataset values')
    ap.add_argument('--overwrite', action='store_true', help='start over if --out holds a different conversion')
    args = ap.parse_args()

    S, rows, sel, betas, name_of, kind_of = make_plan(args.src, args.bodies)
    n_frames = np.array([S['n_frames'][k] for k, _, _ in rows], np.int64)
    offsets = np.concatenate([[0], np.cumsum(n_frames)]).astype(np.int64)
    N, M = int(offsets[-1]), len(rows)
    out = Path(args.out)
    state = out / '.prepare'
    plan = dict(layout=LAYOUT, bodies=[name_of[b] for b in sel], n_motions=M, n_frames=N)
    if (state / 'plan.json').exists() and json.load(open(state / 'plan.json')) != plan:
        if not args.overwrite:
            raise SystemExit(f'{out} holds a different conversion (other --bodies?); pass --overwrite or another --out')
        shutil.rmtree(out)
    fresh = not (state / 'plan.json').exists()
    print(f'{len(sel)} bodies, {M} motions, {N} frames (~{N * BYTES_PER_FRAME / 1e9:.1f} GB) -> {out}', flush=True)
    if fresh:
        free = shutil.disk_usage(out.parent if out.parent.exists() else Path('.')).free
        if free < N * BYTES_PER_FRAME * 1.02:
            raise SystemExit(f'not enough disk space: {free / 1e9:.0f} GB free')
        for d in ('frames', 'motions', 'done'):
            (out / d if d != 'done' else state / d).mkdir(parents=True, exist_ok=True)
        for k, shape in FRAME_SPECS.items():
            open_memmap(out / 'frames' / f'{k}.npy', mode='w+', dtype=np.float32, shape=(N,) + shape).flush()
        json.dump(plan, open(state / 'plan.json', 'w'))

    entries = {b: [] for b in sel}
    for i, (k, b, _) in enumerate(rows):
        entries[b].append((S['clip'][k], k, int(offsets[i])))
    jobs = []
    for b in sel:
        done = state / 'done' / f'{name_of[b]}.json'
        if not done.exists():
            jobs.append((str(out), args.src, b, name_of[b], betas[b], entries[b], S['c_off'], str(done)))
    print(f'{len(sel) - len(jobs)} bodies already done, {len(jobs)} to convert with {args.workers} workers', flush=True)
    t0 = time.time()
    if jobs:
        with Pool(min(args.workers, len(jobs)), initializer=_init, initargs=(S['contact'],)) as pool:
            for i, (body, n, sec) in enumerate(pool.imap_unordered(body_job, jobs)):
                print(f'[{i + 1}/{len(jobs)}] {body}: {n} clips, {sec:.0f} s (elapsed {time.time() - t0:.0f} s)', flush=True)

    write_tables(out, args.src, S, rows, offsets, name_of, kind_of, betas)
    meta = pickle.load(open(TEMPLATE, 'rb'))
    if args.recompute_stats:
        acc, acc2 = np.zeros(3), np.zeros(3)
        for b in sel:
            r = json.load(open(state / 'done' / f'{name_of[b]}.json'))
            acc += r['sum']
            acc2 += r['sum2']
        mean = acc / N
        meta['root_pos_mean'] = mean.astype(np.float32)
        meta['root_pos_std'] = np.sqrt(np.maximum(acc2 / N - mean ** 2, 0)).astype(np.float32)
    meta.update(layout=LAYOUT, joint_num=23, shape_dim=10, n_traj_augs=len(TRAJ_SIGMAS), n_motions=M, n_frames_total=N)
    with open(out / 'meta.pkl', 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'flat_v1 written to {out}: {M} motions, {N} frames ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
