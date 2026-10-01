"""Balanced clip sampling and holdout masks for flat_v1 datasets (plan §4.4).

Strata come from <data>/manifest.csv (written by data/retarget: columns subject, body, kind, ...) when it exists,
otherwise from meta.jsonl (skel_name).  The sampler draws, per epoch, `num_samples` clips: a stratum uniformly,
then a clip uniformly inside it -- so every subject (or subject x body kind) gets the same share regardless of how
many clips it has.  Each DDP rank draws its own independent stream (seed, epoch, rank), which is what
`use_distributed_sampler=False` in train.py expects.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler


def _rank_world():
    r, w = os.environ.get('RANK'), os.environ.get('WORLD_SIZE')
    return (int(r) if r else 0), (int(w) if w else 1)


def motion_table(data_dir):
    """Per-motion metadata rows (dicts) for stratification / holdout: manifest.csv if present, else meta.jsonl."""
    d = Path(data_dir)
    if (d / 'manifest.csv').exists():
        rows = list(csv.DictReader(open(d / 'manifest.csv')))
        rows.sort(key=lambda r: int(r['idx']))
        return rows
    rows = []
    for l in open(d / 'meta.jsonl'):
        if l.strip():
            m = json.loads(l)
            # own-body datasets (own_v1, smoke): the take is the BVH stem and its own source (src = idx), so the
            # take-level holdout of split_v2 applies to them too (docs/paper_eval_plan.md R4)
            rows.append(dict(idx=m['idx'], subject=m['skel_name'], body='', kind='own', style=(m.get('label') or {}).get('style', ''),
                             stem=Path(m['filepath']).stem, src=m['idx']))
    return rows


SPLIT_KEYS = ('bodies', 'pairs', 'takes')


def load_split(split_dir, apply=None):
    """data/eval/split_v2 (scripts/make_split.py) -> the holdout dict `motion_mask_from_cfg` understands:
    bodies (held-out target bodies), pairs (withheld persona x body cells), takes (held-out source stems).

    `apply` selects WHICH of the three clauses leave training; None (the default) = all three, i.e. every run up to
    2026-09-11 is unchanged. The stage-1 codec wave uses `['bodies']`: a reconstruction model may see every take and
    every persona x body cell (the prior is what the held-out columns evaluate), but the 15 held-out bodies stay out
    because the decoder is shape-conditioned -- it is the component that has to render an unseen body, so letting it
    train on those bodies would make the `heldout` column not held out at all. `[]` = train on everything (the V5
    control that measures what the body restriction costs).
    """
    d = Path(split_dir)
    keep = set(SPLIT_KEYS if apply is None else apply)
    for k in keep:
        if k not in SPLIT_KEYS:
            raise ValueError(f'split_apply: unknown clause {k!r} (known: {SPLIT_KEYS})')
    def read(name):
        p = d / f'{name}.csv'
        if not p.exists():
            raise FileNotFoundError(f'{p} missing (scripts/make_split.py)')
        return list(csv.DictReader(l for l in open(p) if not l.startswith('#')))
    out = {'bodies': [int(r['body']) for r in read('heldout_bodies')],
           'pairs': [[r['persona'], int(r['body'])] for r in read('withheld_cells')],
           'takes': [r['stem'] for r in read('heldout_takes')]}
    return {k: (v if k in keep else []) for k, v in out.items()}


def split_hashes(split_dir):
    """sha256 of the four split files, recorded into the run config so check_run_config catches a changed split."""
    import hashlib
    d = Path(split_dir)
    return {n: hashlib.sha256(open(d / f'{n}.csv', 'rb').read()).hexdigest()
            for n in ('heldout_bodies', 'sweep_bodies', 'withheld_cells', 'heldout_takes')}


def motion_mask_from_cfg(data_dir, holdout, split=None, split_apply=None):
    """holdout: None | {bodies: [ids], subjects: [names], pairs: [[subject, body_id], ...], takes: [stems]} and/or
    split: a split_v2 directory (its bodies / pairs / takes are added) -> bool mask over motions or None.
    A take's stem excludes EVERY row derived from it (the own row and all its retargets share the stem).
    split_apply: which of the split's three clauses to apply (see `load_split`); None = all three."""
    if not holdout and not split:
        return None
    holdout = dict(holdout or {})
    if split:
        sp = load_split(split, apply=None if split_apply is None else list(split_apply))
        for k in ('bodies', 'pairs', 'takes'):
            holdout[k] = list(holdout.get(k) or []) + sp[k]
    rows = motion_table(data_dir)
    bodies = {str(b) for b in (holdout.get('bodies') or [])}
    subjects = {str(s) for s in (holdout.get('subjects') or [])}
    pairs = {(str(s), str(b)) for s, b in (holdout.get('pairs') or [])}
    takes = {str(t) for t in (holdout.get('takes') or [])}
    if takes and not all('stem' in r for r in rows):
        raise ValueError('take-level holdout needs a stem per motion (manifest.csv or a meta.jsonl with filepath)')
    mask = np.ones(len(rows), dtype=bool)
    for k, r in enumerate(rows):
        if (r.get('body', '') in bodies or r.get('subject') in subjects or (r.get('subject'), r.get('body', '')) in pairs
                or r.get('stem') in takes):
            mask[k] = False
    return mask


class StratifiedClipSampler(Sampler):
    """Per-epoch random clips balanced over strata (default: subject). Works with DataLoader(sampler=...)."""

    def __init__(self, dataset, by=('subject',), num_samples=None, seed=0):
        self.ds = dataset
        rows = motion_table(dataset.data_dir)
        keys = [tuple(str(rows[m].get(k, '')) for k in by) if m < len(rows) else ('?',) for m in range(dataset.n_motions)]
        strata = {}
        for m, key in enumerate(keys):
            n = int(dataset.clips_per_motion[m])
            if n > 0:
                strata.setdefault(key, []).append((int(dataset.clip_offsets[m]), n))
        self.strata = [np.concatenate([np.arange(o, o + n) for o, n in v]) for v in strata.values()]
        self.keys = list(strata.keys())
        rank, world = _rank_world()
        self.rank, self.world = rank, world
        self.num_samples = int(num_samples) if num_samples else max(1, len(dataset) // world)
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.rank]))
        s = rng.integers(len(self.strata), size=self.num_samples)
        out = np.empty(self.num_samples, dtype=np.int64)
        for k, clips in enumerate(self.strata):
            sel = np.nonzero(s == k)[0]
            if len(sel):
                out[sel] = clips[rng.integers(len(clips), size=len(sel))]
        return iter(out.tolist())

    def __len__(self):
        return self.num_samples
