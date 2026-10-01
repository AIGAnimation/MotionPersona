"""``flat_v1`` clip dataset (see ``data/flat_writer.py`` for the on-disk layout).

Design:
  * no per-clip index table: clip idx -> (motion, start frame) via ``searchsorted`` on cumulative clip counts;
  * frame arrays are opened lazily per process with ``np.load(mmap_mode='r')`` and shared through the OS page
    cache by every DataLoader worker of every DDP rank;
  * ``__getitem__`` does numpy slicing only and returns *raw* clips; every augmentation random draw is made
    here (``AugParams``) but the rotation math itself runs batched on the GPU in ``data/augment.py``.
"""
import os
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, get_worker_info

LAYOUT = 'flat_v1'
TRAJ_PAD = 18              # 3 sigma of the widest trajectory smoothing (sigma 6)

FRAME_ARRAYS = ('rotations', 'root_pos', 'foot_contact', 'traj_pose')
# Optional per-frame audio stream (audio2motion side project, docs/related/audio2motion_log.md, 2026-09-25):
# frames/audio_feat.npy (N, D), pre-extracted offline at the motion frame rate (data/beat2/make_beat2.py).  Only read
# when the dataset is built with audio=True (data.audio); every locomotion dataset / run is untouched.
AUDIO_ARRAY = 'audio_feat'
MOTION_ARRAYS = ('text_feat', 'shape_feat', 'skel_offset')
# Which per-motion array feeds the text token (`data.text_source`): text_feat = CLIP of the fused persona+style
# prompts written by data/make_pkl.py; persona_feat = CLIP of persona-only prompts (scripts/make_persona_feat.py,
# an add-on array) for the split prior whose style enters as a learned token instead (`model.style_cond=embed`).
TEXT_SOURCES = ('text_feat', 'persona_feat')
TEXT_VARIANTS, TEXT_DIM = 3, 512    # prompt variants per motion x CLIP width (zeros when the dataset has no text_feat)
# Style vocabulary = the learned style token's embedding rows, fixed order = data/make_label.py STYLE_CATEGORY
# (pinned by tests/test_style_token.py). t_pose is the reference style: never in a dataset, keeps the table complete.
STYLE_VOCAB = ('angry', 'depressed', 'fear', 'happy', 'neutral', 'bigstep', 'drunk', 'swimming', 'twofootjump', 't_pose')
STYLE_INDEX = {s: i for i, s in enumerate(STYLE_VOCAB)}
# Persona = a closed-set performer ID + three typed attributes (paper revision 2026-09-04): the prior's persona tokens
# (`model.persona_cond=id_attr`, network/latfm.py PersonaEmbed) are learned embedding rows over these FIXED vocabularies,
# so the tables are dataset-independent (a dataset may cover a subset).  SUBJECT_VOCAB = the 44 annotated performers of
# the grid (anonymised performer IDs p01..p44; the ID is the row index); the attribute vocabularies = the distinct
# meta.csv values, sorted.  Age and gender are cohort metadata and never enter the model.  Pinned by
# tests/test_persona_token.py.  The source performer of a motion = manifest.csv `subject` (retargeted rows keep their
# source), else the raw BVH directory name; the attributes come from meta.jsonl label.role/affiliation/dominance.
SUBJECT_VOCAB = ('p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12',
                 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22', 'p23',
                 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p30', 'p31', 'p32', 'p33', 'p34',
                 'p35', 'p36', 'p37', 'p38', 'p39', 'p40', 'p41', 'p42', 'p43', 'p44')
SUBJECT_INDEX = {s: i for i, s in enumerate(SUBJECT_VOCAB)}
ROLE_VOCAB = ('artist', 'athlete', 'child', 'homemaker', 'performer', 'professional', 'retiree', 'student')
AFFILIATION_VOCAB = ('gregarious', 'moderate', 'reserved', 'sociable', 'withdrawn')
DOMINANCE_VOCAB = ('assertive', 'compliant', 'dominant', 'moderate', 'submissive')
ATTR_KEYS = ('role', 'affiliation', 'dominance')
ATTR_VOCABS = (ROLE_VOCAB, AFFILIATION_VOCAB, DOMINANCE_VOCAB)
ATTR_INDEX = tuple({v: i for i, v in enumerate(vocab)} for vocab in ATTR_VOCABS)


def dataset_vocabs(meta):
    """(subject_vocab, style_vocab) of a dataset: `meta.pkl` may carry its own `subject_vocab` / `style_vocab`
    (audio2motion 2026-09-25: BEAT2 speakers / emotions); absent = the fixed locomotion tuples above, so every
    locomotion dataset and checkpoint resolves to exactly SUBJECT_VOCAB / STYLE_VOCAB (never edit those in place:
    docs/related/speech2gesture.md 9.2)."""
    subj = meta.get('subject_vocab') if meta is not None else None
    sty = meta.get('style_vocab') if meta is not None else None
    return (tuple(subj) if subj else SUBJECT_VOCAB), (tuple(sty) if sty else STYLE_VOCAB)


def style_index(style):
    """meta.jsonl label.style -> row of STYLE_VOCAB; unknown styles fail loudly (the token table is fixed)."""
    try:
        return STYLE_INDEX[str(style)]
    except KeyError:
        raise ValueError(f'unknown style {style!r}; STYLE_VOCAB = {STYLE_VOCAB}') from None


def subject_index(name):
    """source performer name -> row of SUBJECT_VOCAB; unknown performers fail loudly (the ID table is fixed)."""
    try:
        return SUBJECT_INDEX[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f'unknown subject {name!r}; SUBJECT_VOCAB = {SUBJECT_VOCAB}') from None


def attr_index(label):
    """meta.jsonl label -> (3,) int64 rows of ROLE/AFFILIATION/DOMINANCE_VOCAB (typed: 'moderate' affiliation and
    'moderate' dominance are different rows); a missing or unknown value fails loudly."""
    out = np.empty(3, dtype=np.int64)
    for j, (key, index) in enumerate(zip(ATTR_KEYS, ATTR_INDEX)):
        v = str(label.get(key, '')).strip().lower()
        if v not in index:
            raise ValueError(f'unknown {key} {v!r}; vocabulary = {ATTR_VOCABS[j]}')
        out[j] = index[v]
    return out


def motion_subjects(data_dir, meta):
    """Source performer of every motion: manifest.csv `subject` (retargeted rows keep their source performer) when the
    dataset has a manifest, else the raw BVH directory name of meta.jsonl filepath (own-body datasets)."""
    manifest = Path(data_dir) / 'manifest.csv'
    if manifest.exists():
        import csv
        rows = sorted(csv.DictReader(open(manifest)), key=lambda r: int(r['idx']))
        if len(rows) != len(meta):
            raise ValueError(f'{manifest}: {len(rows)} rows for {len(meta)} motions')
        return [r['subject'] for r in rows]
    return [Path(r['filepath']).parent.name for r in meta]


@dataclass(frozen=True)
class AugParams:
    """All random choices of one training clip (drawn on the CPU side, applied on the GPU side)."""
    theta: float        # Y-axis rotation augmentation angle in [0, 2*pi)
    traj_aug: int       # trajectory smoothing variant of traj_pose (0 .. n_traj_augs-1)
    smooth_opt: float   # uniform [0,1): < 0.5 raw traj, < 0.75 gaussian sigma 3, else sigma 6
    text_idx: int       # which of the per-motion text features
    mirror: bool = False  # left/right mirror of the clip (datasets built with --no-mirror store no mirrored copies)


def read_meta(data_dir):
    meta = pickle.load(open(Path(data_dir) / 'meta.pkl', 'rb'))
    if meta.get('layout') != LAYOUT:
        raise ValueError(
            f'{data_dir} is not a {LAYOUT} dataset (layout={meta.get("layout")!r}). '
            f'Run data/make_pkl.py, or convert with: python data/flat_writer.py --from-npz-dir <old> -o <new>')
    return meta


def read_meta_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _process_rank():
    for key in ('RANK', 'LOCAL_RANK'):
        if key in os.environ:
            return int(os.environ[key])
    return 0


class LocoDataset(Dataset):
    """Sliding-window clips of ``past_frame + future_frame`` frames over a flat_v1 dataset."""

    def __init__(self, data_dir, past_frame, future_frame, offset_frame, limited_num=None, seed=0, verbose=True, mirror_prob=0.0, motion_mask=None, mirror_skel=False,
                 traj_pad=TRAJ_PAD, text_source='text_feat', attr_shuffle=None, audio=False, audio_lookahead=0):
        self.data_dir = Path(data_dir)
        # traj_pad: frames of root context read beyond the clip before the trajectory smoothing (None = legacy: smooth
        # the bare clip with scipy's 'reflect' boundary, which brakes the last ~2*sigma trajectory frames to a stop --
        # -15 cm at frame 45 for 1 m/s with sigma 6 -- and the model learns to overshoot the trajectory tail)
        self.traj_pad = None if traj_pad is None else int(traj_pad)
        meta = read_meta(self.data_dir)
        self.T_pose = meta['T_pose']
        self.root_pos_mean = np.asarray(meta['root_pos_mean'], dtype=np.float32)
        self.root_pos_std = np.asarray(meta['root_pos_std'], dtype=np.float32)
        self.joint_num = int(meta['joint_num'])
        self.shape_dim = int(meta['shape_dim'])
        self.subject_vocab, self.style_vocab = dataset_vocabs(meta)
        subject_lut = {s: i for i, s in enumerate(self.subject_vocab)}
        style_lut = {s: i for i, s in enumerate(self.style_vocab)}
        self.n_traj_augs = int(meta['n_traj_augs'])
        self.rot_req, self.per_rot_feat = '6d', 6   # features are produced in 6d by data/augment.py

        self.past_frame, self.future_frame = int(past_frame), int(future_frame)
        self.window = self.past_frame + self.future_frame
        self.stride = int(offset_frame)
        if self.stride < 1:
            raise ValueError('offset_frame must be >= 1')
        self.reference_frame_idx = self.past_frame
        self.seed = int(seed)

        self._paths = {k: self.data_dir / 'frames' / f'{k}.npy' for k in FRAME_ARRAYS}
        self._paths.update({k: self.data_dir / 'motions' / f'{k}.npy' for k in MOTION_ARRAYS})
        # audio stream (default off): the clip window plus `audio_lookahead` frames of future audio
        self.audio, self.audio_lookahead = bool(audio), int(audio_lookahead)
        if self.audio:
            self._paths[AUDIO_ARRAY] = self.data_dir / 'frames' / f'{AUDIO_ARRAY}.npy'
            if not self._paths[AUDIO_ARRAY].exists():
                raise FileNotFoundError(f'data.audio=true but {self._paths[AUDIO_ARRAY]} is missing')
            self.audio_dim = int(np.load(self._paths[AUDIO_ARRAY], mmap_mode='r').shape[1])
        self.text_source = str(text_source)
        if self.text_source not in TEXT_SOURCES:
            raise ValueError(f'text_source must be one of {TEXT_SOURCES}, got {text_source!r}')
        self._paths['text_feat'] = self.data_dir / 'motions' / f'{self.text_source}.npy'   # served as item key 'text_feat'
        # motions/text_feat.npy is optional (scripts/prepare_data.py does not write it): only persona_cond=text reads
        # the text token; without the file the loader serves zeros and keeps n_text = TEXT_VARIANTS, so the
        # augmentation draws are those of a dataset with text features
        self.has_text = self._paths['text_feat'].exists()
        if not self.has_text:
            if self.text_source != 'text_feat':
                raise FileNotFoundError(f'{self._paths["text_feat"]} missing')
            del self._paths['text_feat']
            if verbose:
                print(f'{self.data_dir}/motions/text_feat.npy absent: text features are zeros (unused unless model.persona_cond=text)')
        self._mm, self._mm_pid = None, -1
        self._rng, self._rng_pid = None, -1

        self.frame_offsets = np.load(self.data_dir / 'motions' / 'frame_offsets.npy').astype(np.int64)
        self.meta = read_meta_jsonl(self.data_dir / 'meta.jsonl')
        m_all = len(self.meta)
        if m_all + 1 != len(self.frame_offsets):
            raise ValueError('meta.jsonl and frame_offsets.npy disagree on the number of motions')
        self.n_motions = m_all if limited_num is None or int(limited_num) < 0 else min(int(limited_num), m_all)
        self.n_text = int(np.load(self._paths['text_feat'], mmap_mode='r').shape[1]) if self.has_text else TEXT_VARIANTS
        # per-motion style row of STYLE_VOCAB from meta label.style (-1 = unlabeled motion; the style token rejects it)
        def _style_row(st):
            if st not in style_lut:
                raise ValueError(f'unknown style {st!r}; style vocabulary = {self.style_vocab}')
            return style_lut[st]
        self.style_idx = np.array([_style_row(str(r['label']['style'])) if r.get('label') and 'style' in r['label'] else -1
                                   for r in self.meta], dtype=np.int64)
        # per-motion persona rows: subject_idx (SUBJECT_VOCAB) and attr_idx (3,) over ATTR_VOCABS; -1 = not in the
        # vocabulary / no label (the persona tokens reject it). attr_shuffle=<seed> = the shuffled-attribute ablation:
        # every known subject carries ANOTHER subject's attribute triple (a seeded derangement, consistent per subject)
        self.subject_idx = np.array([subject_lut.get(str(s).strip().lower(), -1) for s in motion_subjects(self.data_dir, self.meta)],
                                    dtype=np.int64)
        self.attr_idx = np.full((m_all, 3), -1, dtype=np.int64)
        for i, r in enumerate(self.meta):
            lab = r.get('label') or {}
            if all(k in lab for k in ATTR_KEYS):
                self.attr_idx[i] = attr_index(lab)
        self.attr_shuffle = None if attr_shuffle is None else int(attr_shuffle)
        if self.attr_shuffle is not None:
            # permute among the performers PRESENT in this dataset (a cyclic shift of a seeded order = no fixed point)
            per_subject = {}
            for i in range(m_all):
                if self.subject_idx[i] >= 0 and self.attr_idx[i, 0] >= 0:
                    per_subject.setdefault(int(self.subject_idx[i]), self.attr_idx[i].copy())
            present = np.array(sorted(per_subject), dtype=np.int64)
            if len(present) < 2:
                raise ValueError('data.attr_shuffle needs at least two annotated performers in the dataset')
            order = present[np.random.default_rng(self.attr_shuffle).permutation(len(present))]
            donor = dict(zip(order.tolist(), np.roll(order, 1).tolist()))
            for i in range(m_all):
                s_i = int(self.subject_idx[i])
                if s_i in donor:
                    self.attr_idx[i] = per_subject[donor[s_i]]

        n_frames = np.diff(self.frame_offsets[:self.n_motions + 1])
        self.clips_per_motion = np.maximum(0, (n_frames - self.window) // self.stride + 1).astype(np.int64)
        if motion_mask is not None:                       # holdout / filtering: excluded motions contribute no clips
            mask = np.asarray(motion_mask, dtype=bool)[:self.n_motions]
            self.clips_per_motion[~mask] = 0
        self.mirror_prob = float(mirror_prob)
        # SMPL-X skeletons are left/right asymmetric by 1-4 cm (template joints), so mirroring the motion on the
        # unmirrored skeleton is not a symmetry (feet end up ~1-2 cm off the ground, contact labels drift).
        # mirror_skel=True also mirrors skel_offset (swap sides, negate x) -> the mirrored clip is exact.
        self.mirror_skel = bool(mirror_skel)
        names = list(self.T_pose.names)
        l_names = sorted(n for n in names if 'left' in n.lower()); r_names = sorted(n for n in names if 'right' in n.lower())
        self.mirror_perm = np.arange(len(names))
        for ln, rn in zip(l_names, r_names):
            self.mirror_perm[names.index(ln)] = names.index(rn); self.mirror_perm[names.index(rn)] = names.index(ln)
        self.clip_offsets = np.concatenate([[0], np.cumsum(self.clips_per_motion)]).astype(np.int64)

        if verbose:
            used = self.clips_per_motion > 0                                   # long enough AND not held out
            print('Dataset loaded: Including %d motion sequences and %d frames (%d motions in the file%s).'
                  % (int(used.sum()), int(n_frames[used].sum()), m_all,
                     '' if motion_mask is None else ', %d excluded by the holdout' % int((~np.asarray(motion_mask, bool)[:self.n_motions]).sum())))
            print('We slice the dataset into %d clips, each clip has %d frames.' % (len(self), self.window))

    # ------------------------------------------------------------------ indexing
    def __len__(self):
        return int(self.clip_offsets[-1])

    def clip_to_motion_start(self, idx):
        """clip index (int or int array) -> (motion id, GLOBAL start frame)."""
        idx = np.asarray(idx, dtype=np.int64)
        if np.any(idx < 0) or np.any(idx >= len(self)):
            raise IndexError('clip index out of range')
        m = np.searchsorted(self.clip_offsets, idx, side='right') - 1
        local = (idx - self.clip_offsets[m]) * self.stride
        return m, self.frame_offsets[m] + local

    def file_path(self, motion_idx):
        return self.meta[int(motion_idx)]['filepath']

    def skel_name(self, motion_idx):
        return self.meta[int(motion_idx)]['skel_name']

    # ------------------------------------------------------------------ per-process state
    def _arrays(self):
        pid = os.getpid()
        if self._mm is None or self._mm_pid != pid:
            self._mm = {k: np.load(p, mmap_mode='r') for k, p in self._paths.items()}
            self._mm_pid = pid
        return self._mm

    def _rng_for_process(self):
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            info = get_worker_info()
            worker_seed = int(info.seed) if info is not None else self.seed
            self._rng = np.random.default_rng(np.random.SeedSequence([worker_seed, _process_rank(), self.seed]))
            self._rng_pid = pid
        return self._rng

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_mm'], state['_mm_pid'] = None, -1
        state['_rng'], state['_rng_pid'] = None, -1
        return state

    def prewarm(self, chunk_bytes=64 << 20):
        """Sequentially read the frame arrays once so the OS page cache is hot before random access."""
        for k in FRAME_ARRAYS + ((AUDIO_ARRAY,) if self.audio else ()):
            with open(self._paths[k], 'rb') as f:
                while f.read(chunk_bytes):
                    pass

    # ------------------------------------------------------------------ items
    def sample_aug(self, rng, mirror=None):
        """One clip's augmentation draws. ``mirror=False`` skips the mirror draw entirely (same stream as a
        ``mirror_prob=0`` dataset), which is how the fixed export batch stays independent of ``mirror_prob``."""
        theta = float(rng.uniform(0, 2 * np.pi))
        traj_aug = int(rng.integers(self.n_traj_augs))
        smooth_opt = float(rng.random())
        text_idx = int(rng.integers(self.n_text))
        if mirror is None:
            mirror = self.mirror_prob > 0 and rng.random() < self.mirror_prob
        return AugParams(theta, traj_aug, smooth_opt, text_idx, mirror=bool(mirror))

    def export_rng(self):
        """The stream scripts/export_samples.py sees: a fresh main-process (rank 0) dataset RNG."""
        return np.random.default_rng(np.random.SeedSequence([self.seed, 0, self.seed]))

    def get_raw_item(self, idx, aug):
        """Deterministic clip assembly (no randomness besides ``aug``). Returns numpy arrays, never memmap views."""
        A = self._arrays()
        m, f0 = self.clip_to_motion_start(int(idx))
        m, f0 = int(m), int(f0)
        sl = slice(f0, f0 + self.window)
        r = self.reference_frame_idx

        rotations = np.array(A['rotations'][sl], dtype=np.float32)              # (T, J, 4)
        root_pos = np.array(A['root_pos'][sl], dtype=np.float32)                # (T, 3)
        foot_contact = np.array(A['foot_contact'][sl], dtype=np.float32)        # (T, 2)
        traj_quat = np.array(A['traj_pose'][sl, aug.traj_aug], dtype=np.float32)  # (T, 4)
        sigma = 0 if aug.smooth_opt < 0.5 else (3 if aug.smooth_opt < 0.75 else 6)
        traj_xz = None
        if sigma and self.traj_pad is not None:
            # smooth the pelvis path with context from the surrounding motion so the clip edges are not boundaries
            pad = self.traj_pad
            p0, p1 = max(int(self.frame_offsets[m]), f0 - pad), min(int(self.frame_offsets[m + 1]), f0 + self.window + pad)
            ctx = np.array(A['root_pos'][p0:p1, [0, 2]], dtype=np.float32)
            traj_xz = gaussian_filter1d(ctx, sigma, axis=0, mode='nearest')[f0 - p0:f0 - p0 + self.window]
        if aug.mirror:
            # same operator as utils/motion_processing.mirror on a whole motion (what make_pkl used to store):
            # swap left/right joints, negate the y/z quaternion components (root sign flip too), negate x
            rotations = rotations[:, self.mirror_perm]
            rotations[:, :, 2:] *= -1
            rotations[:, 0] *= -1
            root_pos[:, 0] *= -1
            if traj_xz is not None:
                traj_xz[:, 0] *= -1
            foot_contact = foot_contact[:, ::-1].copy()
            traj_quat = traj_quat.copy(); traj_quat[:, 2:] *= -1
        root_pos[:, [0, 2]] -= root_pos[r - 1:r, [0, 2]]

        if traj_xz is None:                                     # raw path, or legacy bare-clip smoothing ('reflect' boundary)
            traj_xz = root_pos[:, [0, 2]].copy()
            if sigma:
                traj_xz = gaussian_filter1d(traj_xz, sigma, axis=0)
        traj_xz -= traj_xz[r - 1:r]

        skel_offset = np.array(A['skel_offset'][m], dtype=np.float32)
        if aug.mirror and self.mirror_skel:
            skel_offset = skel_offset[self.mirror_perm].copy()
            skel_offset[:, 0] *= -1

        theta = aug.theta
        aug_trig = np.array([np.cos(theta * 0.5), np.sin(theta * 0.5), np.cos(theta), np.sin(theta)], dtype=np.float32)

        out = {
            'rotations': rotations,
            'root_pos': root_pos,
            'foot_contact': foot_contact,
            'traj_quat': np.ascontiguousarray(traj_quat[r:]),
            'traj_xz': np.ascontiguousarray(traj_xz[r:]),
            'aug_trig': aug_trig,
            'text_feat': (np.array(A['text_feat'][m, aug.text_idx], dtype=np.float32) if self.has_text  # motions/<text_source>.npy
                          else np.zeros(TEXT_DIM, np.float32)),
            'shape_feat': np.array(A['shape_feat'][m], dtype=np.float32),
            'skel_offset': skel_offset,
            'style_idx': np.int64(self.style_idx[m]),
            'subject_idx': np.int64(self.subject_idx[m]),
            'attr_idx': self.attr_idx[m].copy(),
            'motion_idx': np.int64(m),
            'clip_start': np.int64(f0 - self.frame_offsets[m]),
        }
        if self.audio:
            # window + lookahead frames of audio; past the end of the motion the last frame is repeated
            m_end = int(self.frame_offsets[m + 1])
            n_a = self.window + self.audio_lookahead
            a = np.array(A[AUDIO_ARRAY][f0:min(m_end, f0 + n_a)], dtype=np.float32)
            if len(a) < n_a:
                a = np.concatenate([a, np.repeat(a[-1:], n_a - len(a), 0)], 0)
            out['audio_feat'] = a                                                  # (window + lookahead, D)
        return out

    def __getitem__(self, idx):
        return self.get_raw_item(int(idx), self.sample_aug(self._rng_for_process()))
