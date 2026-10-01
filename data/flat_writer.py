"""Writer: a directory of per-motion npz files -> the ``flat_v1`` memmap layout.

Layout written under ``<out>/``::

    meta.pkl                      T_pose, parents, names, root_pos_mean/std, joint_num, shape_dim,
                                  n_traj_augs, layout='flat_v1', n_motions, n_frames_total
    meta.jsonl                    one line per motion (idx, n_frames, frame_offset, filepath, skel_name,
                                  mirror, text_raw, label), in motion-id order
    frames/rotations.npy          (N, J, 4)  float32   wxyz local quaternions, all motions concatenated
    frames/root_pos.npy           (N, 3)     float32
    frames/foot_contact.npy       (N, 2)     float32
    frames/traj_pose.npy          (N, A, 4)  float32   A = number of trajectory smoothing variants
    motions/frame_offsets.npy     (M+1,)     int64     motion i owns frames [offsets[i], offsets[i+1])
    motions/text_feat.npy         (M, 3, 512) float32   CLIP of the 3 fused "persona; style" prompts (text_raw)
    motions/shape_feat.npy        (M, 10)    float32
    motions/skel_offset.npy       (M, J, 3)  float32
    motions/persona_feat.npy      (M, 3, 512) float32   OPTIONAL add-on: CLIP of
                                  the 3 persona-only prompts, served as 'text_feat' under data.text_source=persona_feat

The source directory is a per-motion layout (``meta.pkl`` + ``meta.jsonl`` + one npz per motion with the keys
``local_joint_rotations``, ``global_root_positions``, ``foot_contact``, ``traj_pose``, ``text_feat``,
``shape_feat``, ``skel_offset``)::

    python data/flat_writer.py --from-npz-dir <npz_dir> -o <out_dir>
"""
import sys
sys.path.append('./')

import json
import pickle
import shutil
import argparse
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

LAYOUT = 'flat_v1'

FRAME_ARRAYS = ('rotations', 'root_pos', 'foot_contact', 'traj_pose')
MOTION_ARRAYS = ('text_feat', 'shape_feat', 'skel_offset')


def read_meta_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def npz_path_of(src_dir, entry):
    rel = entry.get('motion_feature_path') or f"{int(entry['idx']):06d}.npz"
    return Path(src_dir) / rel


def flatten_npz_dir(src_dir, out_dir, t_pose=None, remove_src=False, verbose=True, reuse_stats=True):
    """Stream every per-motion npz under ``src_dir`` into the flat layout at ``out_dir``.

    Never holds more than one motion in RAM. ``root_pos_mean/std`` are accumulated in float64 over all
    frames of all motions (population std), which matches float32 statistics to ~1e-5.
    When ``t_pose is None`` (read from ``src_dir/meta.pkl``) and ``reuse_stats`` is set, the source dir's
    ``root_pos_mean/std`` are copied verbatim instead, so that features produced from the converted dataset
    are bit-identical to those of the source dir.
    """
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    old_stats = None
    metas = sorted(read_meta_jsonl(src_dir / 'meta.jsonl'), key=lambda m: int(m['idx']))
    if [int(m['idx']) for m in metas] != list(range(len(metas))):
        raise ValueError('meta.jsonl idx must be contiguous 0..M-1')
    if not metas:
        raise ValueError(f'no motions listed in {src_dir / "meta.jsonl"}')
    if t_pose is None:
        src_meta = pickle.load(open(src_dir / 'meta.pkl', 'rb'))
        t_pose = src_meta['T_pose']
        if reuse_stats and 'root_pos_mean' in src_meta:
            old_stats = (np.asarray(src_meta['root_pos_mean'], dtype=np.float32),
                         np.asarray(src_meta['root_pos_std'], dtype=np.float32))

    n_frames = np.array([int(m['n_frames']) for m in metas], dtype=np.int64)
    frame_offsets = np.concatenate([[0], np.cumsum(n_frames)]).astype(np.int64)
    N, M = int(frame_offsets[-1]), len(metas)

    first = np.load(npz_path_of(src_dir, metas[0]))
    J = int(first['local_joint_rotations'].shape[1])
    n_aug = int(first['traj_pose'].shape[0])
    n_text, text_dim = (int(v) for v in first['text_feat'].shape)
    shape_dim = int(first['shape_feat'].shape[0])

    frames_dir, motions_dir = out_dir / 'frames', out_dir / 'motions'
    frames_dir.mkdir(parents=True, exist_ok=True)
    motions_dir.mkdir(parents=True, exist_ok=True)

    mm = {
        'rotations': open_memmap(frames_dir / 'rotations.npy', mode='w+', dtype=np.float32, shape=(N, J, 4)),
        'root_pos': open_memmap(frames_dir / 'root_pos.npy', mode='w+', dtype=np.float32, shape=(N, 3)),
        'foot_contact': open_memmap(frames_dir / 'foot_contact.npy', mode='w+', dtype=np.float32, shape=(N, 2)),
        'traj_pose': open_memmap(frames_dir / 'traj_pose.npy', mode='w+', dtype=np.float32, shape=(N, n_aug, 4)),
        'text_feat': open_memmap(motions_dir / 'text_feat.npy', mode='w+', dtype=np.float32, shape=(M, n_text, text_dim)),
        'shape_feat': open_memmap(motions_dir / 'shape_feat.npy', mode='w+', dtype=np.float32, shape=(M, shape_dim)),
        'skel_offset': open_memmap(motions_dir / 'skel_offset.npy', mode='w+', dtype=np.float32, shape=(M, J, 3)),
    }

    acc = np.zeros(3, dtype=np.float64)
    acc2 = np.zeros(3, dtype=np.float64)
    iterator = tqdm(metas, desc='Flattening') if verbose else metas
    for i, m in enumerate(iterator):
        z = np.load(npz_path_of(src_dir, m))
        s, e = int(frame_offsets[i]), int(frame_offsets[i + 1])
        rot = z['local_joint_rotations']
        if rot.shape[0] != e - s:
            raise ValueError(f'motion {i}: meta n_frames={e - s} but npz has {rot.shape[0]} frames')
        if rot.shape[1] != J:
            raise ValueError(f'motion {i}: joint count {rot.shape[1]} != {J}')
        root = z['global_root_positions'].astype(np.float32, copy=False)
        mm['rotations'][s:e] = rot.astype(np.float32, copy=False)
        mm['root_pos'][s:e] = root
        mm['foot_contact'][s:e] = z['foot_contact'].astype(np.float32, copy=False)
        mm['traj_pose'][s:e] = np.ascontiguousarray(z['traj_pose'].transpose(1, 0, 2)).astype(np.float32, copy=False)
        mm['text_feat'][i] = z['text_feat'].astype(np.float32, copy=False)
        mm['shape_feat'][i] = z['shape_feat'].astype(np.float32, copy=False)
        mm['skel_offset'][i] = z['skel_offset'].astype(np.float32, copy=False)
        root64 = root.astype(np.float64)
        acc += root64.sum(axis=0)
        acc2 += (root64 ** 2).sum(axis=0)

    for arr in mm.values():
        arr.flush()
    del mm

    mean = acc / max(N, 1)
    var = np.maximum(acc2 / max(N, 1) - mean ** 2, 0.0)
    root_pos_mean = mean.astype(np.float32)
    root_pos_std = np.sqrt(var).astype(np.float32)
    if old_stats is not None:
        root_pos_mean, root_pos_std = old_stats

    np.save(motions_dir / 'frame_offsets.npy', frame_offsets)

    with open(out_dir / 'meta.jsonl', 'w') as f:
        for i, m in enumerate(metas):
            filepath = str(m['filepath'])
            entry = {
                'idx': i,
                'n_frames': int(n_frames[i]),
                'frame_offset': int(frame_offsets[i]),
                'filepath': filepath,
                'skel_name': m['skel_name'],
                'mirror': bool(m['mirror']) if 'mirror' in m else filepath.endswith('.mirror.bvh'),
                'text_raw': m.get('text_raw'),
                'label': m.get('label'),
            }
            f.write(json.dumps(entry) + '\n')

    meta_pkl = {
        'layout': LAYOUT,
        'T_pose': t_pose,
        'parents': t_pose.parents,
        'names': t_pose.names,
        'root_pos_mean': root_pos_mean,
        'root_pos_std': root_pos_std,
        'joint_num': J,
        'shape_dim': shape_dim,
        'n_traj_augs': n_aug,
        'n_motions': M,
        'n_frames_total': N,
    }
    with open(out_dir / 'meta.pkl', 'wb') as f:
        pickle.dump(meta_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)

    if remove_src:
        shutil.rmtree(src_dir)
    if verbose:
        print(f'flat_v1 written to {out_dir}: {M} motions, {N} frames, J={J}, traj_augs={n_aug}')
    return meta_pkl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a per-motion npz dataset dir into the flat_v1 memmap layout')
    parser.add_argument('--from-npz-dir', required=True, type=str, help='per-motion npz dir (meta.pkl + meta.jsonl + motions/*.npz)')
    parser.add_argument('-o', '--output', required=True, type=str, help='output dir')
    parser.add_argument('--remove-src', action='store_true', help='delete the source dir after a successful conversion')
    args = parser.parse_args()
    flatten_npz_dir(args.from_npz_dir, args.output, remove_src=args.remove_src)
