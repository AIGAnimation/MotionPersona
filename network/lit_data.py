import lightning as L
from torch.utils.data import DataLoader

from data.loco_dataset import LocoDataset, read_meta, TRAJ_PAD


class LocoDataModule(L.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()
        self.cfg = data_cfg
        self.train_set = None

    def _make_dataset(self, verbose):
        from data.samplers import motion_mask_from_cfg
        mask = motion_mask_from_cfg(self.cfg.path, self.cfg.get('holdout', None), split=self.cfg.get('split', None),
                                    split_apply=self.cfg.get('split_apply', None))
        return LocoDataset(self.cfg.path, self.cfg.past_frame, self.cfg.future_frame, self.cfg.offset_frame,
                           limited_num=self.cfg.limited_num, seed=self.cfg.seed, verbose=verbose,
                           mirror_prob=float(self.cfg.get('mirror_prob', 0.0)), motion_mask=mask,
                           mirror_skel=bool(self.cfg.get('mirror_skel', False)),
                           traj_pad=self.cfg.get('traj_pad', TRAJ_PAD),
                           text_source=str(self.cfg.get('text_source', 'text_feat')),
                           attr_shuffle=self.cfg.get('attr_shuffle', None),
                           audio=bool(self.cfg.get('audio', False)),                  # speech gestures (default off)
                           audio_lookahead=int(self.cfg.get('audio_lookahead', 0)))

    def prepare_data(self):
        # local-rank-0 only: pull the frame arrays through the page cache once before random access starts
        if self.cfg.prewarm:
            self._make_dataset(verbose=False).prewarm()

    def setup(self, stage=None):
        if self.train_set is None:
            self.train_set = self._make_dataset(verbose=True)

    def train_dataloader(self):
        workers = int(self.cfg.workers)
        sampler = None
        if str(self.cfg.get('sampler', 'uniform')) == 'balanced':
            from data.samplers import StratifiedClipSampler
            sampler = StratifiedClipSampler(self.train_set, by=list(self.cfg.get('balance_by', ['subject'])), seed=int(self.cfg.seed))
        return DataLoader(
            self.train_set,
            batch_size=int(self.cfg.batch_size),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=bool(self.cfg.pin_memory),
            drop_last=bool(self.cfg.drop_last),
            prefetch_factor=int(self.cfg.prefetch_factor) if workers > 0 else None,
        )


__all__ = ['LocoDataModule', 'read_meta']
