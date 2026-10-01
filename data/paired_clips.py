"""Paired clip sampling for the shape-invariant VAE variant (diffusion=vae_inv, network/vae.py VAEInvModule).

The cross dataset is FRAME-ALIGNED: the same source clip retargeted to different bodies has identical
length and timing, and the writer stores label.source_idx per motion.  A pair = (clip of motion A,
the SAME local clip of a partner motion B with the same source_idx), drawn with ONE shared AugParams --
identical mirror/rotation/text draws, otherwise the z-invariance loss would learn augmentation noise.

PairedLocoDataModule keeps `self.train_set` = the plain LocoDataset (so build_export_batch and the whole
export harness keep working on single clips) and only swaps the train_dataloader for the paired dataset.
data.sampler=balanced is not supported here (paired experiments use uniform shuffling).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.loco_dataset import AugParams
from network.lit_data import LocoDataModule


class PairedClipDataset(Dataset):
    def __init__(self, base, seed=0):
        self.base = base
        self.seed = int(seed)
        self._rng = None
        meta = [json.loads(l) for l in open(Path(base.data_dir) / 'meta.jsonl')][:base.n_motions]
        groups = defaultdict(list)
        for m, row in enumerate(meta):
            src = (row.get('label') or {}).get('source_idx')
            if src is not None and base.clips_per_motion[m] > 0:
                groups[src].append(m)
        self.partners = {m: [x for x in g if x != m] for g in groups.values() for m in g}
        n_paired = sum(1 for v in self.partners.values() if v)
        print(f'PairedClipDataset: {n_paired}/{base.n_motions} motions have frame-aligned partners'
              + ('' if n_paired else ' -- FALLING BACK to self-pairs (no label.source_idx?)'))

    def __len__(self):
        return len(self.base)

    def _get_rng(self):
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            self._rng = np.random.RandomState(self.seed * 1000 + (info.id if info else 0) + 1)
        return self._rng

    def __getitem__(self, idx):
        rng = self._get_rng()
        base = self.base
        m = int(np.searchsorted(base.clip_offsets, idx, side='right') - 1)
        local = int(idx - base.clip_offsets[m])
        partners = self.partners.get(m) or [m]
        m2 = int(partners[rng.randint(len(partners))])
        j = int(base.clip_offsets[m2]) + local
        aug = AugParams(float(rng.uniform(0, 2 * np.pi)), int(rng.randint(getattr(base, 'n_traj_augs', 2))),
                        float(rng.rand()), int(rng.randint(getattr(base, 'n_text', 3))),
                        mirror=bool(rng.rand() < float(getattr(base, 'mirror_prob', 0.0))))
        return {'a': base.get_raw_item(idx, aug), 'b': base.get_raw_item(j, aug)}


class PairedLocoDataModule(LocoDataModule):
    def train_dataloader(self):
        assert str(self.cfg.get('sampler', 'uniform')) == 'uniform', 'paired_data supports data.sampler=uniform only'
        workers = int(self.cfg.workers)
        paired = PairedClipDataset(self.train_set, seed=int(self.cfg.seed))
        return DataLoader(paired, batch_size=int(self.cfg.batch_size), shuffle=True, num_workers=workers,
                          persistent_workers=workers > 0, pin_memory=bool(self.cfg.pin_memory),
                          drop_last=bool(self.cfg.drop_last),
                          prefetch_factor=int(self.cfg.prefetch_factor) if workers > 0 else None)
