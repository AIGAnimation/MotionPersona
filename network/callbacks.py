"""Training callbacks: EMA, the epoch-end monitor (best/periodic checkpoints, log.txt/TensorBoard, BVH export),
and the always-on sanity gates (cross-rank DDP checks, finite-loss guard)."""
import contextlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import lightning as L
from torch.utils.data import Subset, default_collate
from torch_ema import ExponentialMovingAverage

from network.checkpoint import load_weights
from utils.foot_lock import measure, summarize

LOG = logging.getLogger('persona')


class EMACallback(L.Callback):
    """torch-ema shadow of ``pl_module.model``; updated after every optimizer step, saved in the checkpoint."""

    def __init__(self, decay=0.995):
        self.decay = float(decay)
        self.ema = None
        self._pending = None

    def on_fit_start(self, trainer, pl_module):
        if self.ema is None:
            self.ema = ExponentialMovingAverage(pl_module.model.parameters(), decay=self.decay)
            if self._pending is not None:
                self.ema.load_state_dict(self._pending)
                self._pending = None
        self.ema.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ema.update()

    def average_parameters(self):
        return self.ema.average_parameters() if self.ema is not None else contextlib.nullcontext()

    def state_dict(self):
        return self.ema.state_dict() if self.ema is not None else {}

    def load_state_dict(self, state_dict):
        if state_dict:
            self._pending = state_dict


def build_export_batch(dataset, pl_module, num_samples, seed):
    """The fixed clips exported during a run: `num_samples` indices from RandomState(seed), featurized once.

    Augmentation comes from a private copy of the dataset's export stream (`LocoDataset.export_rng`) with the
    mirror draw skipped, so the batch is the SAME unmirrored clips (the true retargeted data) in training and in
    scripts/export_samples.py, whatever `data.mirror_prob` is, and nothing here consumes the dataset's own RNG.
    """
    n = min(int(num_samples), len(dataset))
    idx = np.random.RandomState(int(seed)).randint(0, len(dataset), n)
    while isinstance(dataset, Subset):                    # tests wrap the train set in a Subset
        idx, dataset = np.asarray(dataset.indices)[idx], dataset.dataset
    rng = dataset.export_rng()
    raws = [dataset.get_raw_item(int(i), dataset.sample_aug(rng, mirror=False)) for i in idx]
    return pl_module.featurize(default_collate(raws))


def gt_foot_lock_reference(pl_module, gt, skeletons, fl_cfg):
    """``utils.foot_lock.measure`` of the true clips (skating under the same contact detector), for summary.json."""
    from utils import nn_transforms
    rotations = nn_transforms.repr6d2quat(gt[:, :, :-2]).cpu().numpy()
    root = (gt[:, :, -2, :3] * pl_module.root_pos_std + pl_module.root_pos_mean).cpu().numpy()
    skel = skeletons.cpu().numpy()
    tpl = pl_module.T_pose
    scale = float(tpl.scaling_factor) if tpl.scaling_factor else 1.0
    return [measure(rotations[i], root[i] / scale, skel[i] / scale, tpl.parents, tpl.names, fl_cfg)[0] for i in range(gt.shape[0])]


def export_samples(pl_module, batch, out_dir, tag, export_cfg, epoch=-1, ema_ctx=None, noise_seed=None):
    """Write BVH samples + summary.json for one featurized batch.

    motion_i.gt.bvh      the true clip (K history frames + future)
    motion_i.pred.bvh    history + single-pass x0 prediction (random t) of the future  = what a consumer plays
    motion_i.sample.bvh  history + full DDPM sampling loop of the future                = what a consumer plays
    motion_i.*_raw.bvh   (export.raw_prefix) the model's own output for all frames, history included
    The seam metrics in summary.json are computed on exactly the pred/sample sequences written.
    ``noise_seed`` seeds the sampling-loop generator only (default export.seed, which also picks the batch):
    same clips, a different noise draw. The generator lives on the module's device, so CPU and GPU exports of the
    same weights are different draws; a single 16-clip draw swings the median skate/IoU by ~0.07 at 2 steps.
    """
    from scripts.verify_samples import check_bvh_dir, pinned, seam_metrics
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Lightning tears the model down to CPU *before* on_fit_end under DDP; follow the module's current device.
    batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
    cond = batch['conditions']
    skel = cond['skel_offset']
    K = pl_module.past_frame
    was_training = pl_module.training
    pl_module.eval()
    ctx = ema_ctx if ema_ctx is not None else contextlib.nullcontext()
    raw_prefix = bool(export_cfg.get('raw_prefix', False)) if hasattr(export_cfg, 'get') else bool(getattr(export_cfg, 'raw_prefix', False))
    fl_cfg = export_cfg.get('foot_lock', None) if hasattr(export_cfg, 'get') else getattr(export_cfg, 'foot_lock', None)
    fl_enabled = bool(fl_cfg is not None and fl_cfg['enabled'])
    metrics, foot_lock = {}, {}
    with ctx, torch.no_grad():
        def m(feat):
            return seam_metrics(feat, skel, pl_module.skel_parents, pl_module.root_pos_mean, pl_module.root_pos_std, K,
                                pl_module.loss_flags.rot_req)
        gt, pred_raw = pl_module.x0_at_random_t(batch)
        pred = pinned(gt, pred_raw, K)
        pl_module.export_bvh(gt, skel, str(out_dir), 'gt')
        infos = pl_module.export_bvh(pred, skel, str(out_dir), 'pred', foot_lock=fl_cfg)
        metrics['gt'] = m(gt)
        metrics['pred'] = m(pred)
        if fl_enabled:
            gt_infos = gt_foot_lock_reference(pl_module, gt, skel, fl_cfg)
            foot_lock['pred'] = summarize(infos, gt_infos)
        if raw_prefix:
            pl_module.export_bvh(pred_raw, skel, str(out_dir), 'pred_raw')
        if export_cfg.sample:
            gen = torch.Generator(device=pl_module.device).manual_seed(int(export_cfg.seed if noise_seed is None else noise_seed))
            _, smp_raw = pl_module.sample(batch, generator=gen)
            smp = pinned(gt, smp_raw, K)
            infos = pl_module.export_bvh(smp, skel, str(out_dir), 'sample', foot_lock=fl_cfg)
            metrics['sample'] = m(smp)
            if fl_enabled:
                foot_lock['sample'] = summarize(infos, gt_infos)
            if raw_prefix:
                pl_module.export_bvh(smp_raw, skel, str(out_dir), 'sample_raw')
    pl_module.train(was_training)
    report = {'epoch': int(epoch), 'tag': tag, 'metrics': metrics, 'prefix_frames': K,
              'bvh': check_bvh_dir(str(out_dir), pl_module.clip_len)}
    if fl_enabled:
        report['foot_lock'] = foot_lock          # skating (toe xz-speed on contact frames, cm/frame) before/after, gt reference
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(report, f, indent=2)
    return report


class TrainMonitor(L.Callback):
    """Run bookkeeping on top of Lightning's synced epoch metrics.

    * running-loss lines every ``log_interval`` batches, one summary line per epoch (log.txt, rank 0)
    * TensorBoard scalars ``train/<loss>`` at x = epoch
    * ``best.ckpt`` when ``epoch > best_after and loss < best`` (best tracked from epoch 0)
    * ``weights_<epoch>.ckpt`` + BVH export every ``save_freq`` epochs, ``last.ckpt`` every epoch
    * exports: ``samples/init`` before training, ``samples/best`` after training (best weights)
    """

    def __init__(self, run_dir, cfg, ema_callback=None):
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.ema_callback = ema_callback
        self.best_loss = 1e10
        self.export_batch = None
        self._running = {}
        self._running_n = 0
        self.log_interval = None

    # ------------------------------------------------------------------ helpers
    def _fmt(self, losses):
        return ', '.join(f'{k}: {v:.6f}' for k, v in losses.items())

    def _build_export_batch(self, trainer, pl_module):
        self.export_batch = build_export_batch(trainer.datamodule.train_set, pl_module, self.cfg.export.num_samples, self.cfg.export.seed)

    def _export(self, trainer, pl_module, tag):
        if not trainer.is_global_zero:
            return
        out_dir = self.run_dir / 'samples' / tag
        ema_ctx = self.ema_callback.average_parameters() if (self.cfg.trainer.ema.use_for_eval and self.ema_callback) else None
        report = export_samples(pl_module, self.export_batch, out_dir, tag, self.cfg.export, trainer.current_epoch, ema_ctx)
        metrics = report['metrics']
        seam = ' | '.join(f'{k} seam_pos_jump={v["seam_pos_jump"]:.4f} seam_vel_jump={v["seam_vel_jump"]:.4f}' for k, v in metrics.items())
        if report.get('foot_lock'):
            fl = report['foot_lock']
            seam += ' | foot-lock skate ' + ' '.join(f'{k}={v["skate_in"]:.3f}->{v["skate_out"]:.3f}' for k, v in fl.items() if v['skate_in'] is not None)
            gt_ref = next((v.get('skate_gt') for v in fl.values() if v.get('skate_gt') is not None), None)
            if gt_ref is not None:
                seam += f' gt={gt_ref:.3f}'
        LOG.info(f'Evaluate sampling {out_dir} at epoch {trainer.current_epoch} | {seam} | bvh ok={report["bvh"]["ok"]}')

    # ------------------------------------------------------------------ hooks
    def on_train_start(self, trainer, pl_module):
        self._build_export_batch(trainer, pl_module)
        num_batches = trainer.num_training_batches
        li = self.cfg.trainer.log_interval
        self.log_interval = int(li) if li else max(1, int(num_batches) // 50)
        if trainer.is_global_zero:
            LOG.info('Train with %d epochs, %d batches by %d batch_size' % (trainer.max_epochs, num_batches, self.cfg.data.batch_size))
            if trainer.current_epoch == 0:
                self._export(trainer, pl_module, 'init')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        losses = outputs.get('losses', {}) if isinstance(outputs, dict) else {}
        for k, v in losses.items():
            self._running[k] = self._running.get(k, 0.0) + float(v)
        self._running_n += 1
        if trainer.is_global_zero and (batch_idx + 1) % self.log_interval == 0:
            mean = {k: v / self._running_n for k, v in self._running.items()}
            LOG.info(f'  Epoch {trainer.current_epoch} [{batch_idx + 1}/{trainer.num_training_batches}] | {self._fmt(mean)}')
            self._running, self._running_n = {}, 0

    def on_train_epoch_end(self, trainer, pl_module):
        self._running, self._running_n = {}, 0
        metrics = {k[len('train/'):]: float(v) for k, v in trainer.callback_metrics.items() if k.startswith('train/')}
        epoch = trainer.current_epoch
        loss = metrics.get('loss', float('nan'))
        tr = self.cfg.trainer

        if epoch > int(tr.best_after) and loss < self.best_loss:
            trainer.save_checkpoint(str(self.run_dir / 'best.ckpt'))
            if trainer.is_global_zero:
                LOG.info(f'Saved checkpoint: {self.run_dir / "best.ckpt"}')
        if loss < self.best_loss:
            self.best_loss = loss

        if trainer.is_global_zero:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            LOG.info(f'Epoch {epoch}/{trainer.max_epochs} | {self._fmt(metrics)} | best: {self.best_loss:.6f} | lr: {lr:.3e}')
            tb = trainer.logger.experiment if trainer.logger is not None else None
            if tb is not None:
                for k, v in metrics.items():
                    tb.add_scalar(f'train/{k}', v, epoch)
                tb.add_scalar('train/lr', lr, epoch)

        if epoch > 0 and int(tr.save_freq) > 0 and epoch % int(tr.save_freq) == 0:
            trainer.save_checkpoint(str(self.run_dir / f'weights_{epoch}.ckpt'))
            if trainer.is_global_zero:
                LOG.info(f'Saved checkpoint: {self.run_dir / f"weights_{epoch}.ckpt"}')
            self._export(trainer, pl_module, f'epoch{epoch}')

        if tr.save_last:
            trainer.save_checkpoint(str(self.run_dir / 'last.ckpt'))

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        best = self.run_dir / 'best.ckpt'
        if best.exists():
            load_weights(pl_module.model, str(best), map_location=pl_module.device)
            LOG.info(f'Load checkpoint from {best} for the final sample export')
            self._export(trainer, pl_module, 'best')
        else:
            LOG.info('No best.ckpt written (training shorter than best_after epochs); exporting last weights instead')
            self._export(trainer, pl_module, 'last')

    def state_dict(self):
        return {'best_loss': self.best_loss}

    def load_state_dict(self, state_dict):
        self.best_loss = float(state_dict.get('best_loss', 1e10))


class DDPSanityCallback(L.Callback):
    """DDP sanity gate: every epoch, assert the first batch's clips are disjoint across ranks and that all ranks hold
    bit-identical parameters after the epoch (i.e. gradients really were all-reduced)."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.world_size < 2 or batch_idx != 0:
            return
        cond = batch['conditions']
        keys = cond['motion_idx'].to(torch.int64) * (1 << 32) + cond['clip_start'].to(torch.int64)
        gathered = pl_module.all_gather(keys)
        sets = [set(g.tolist()) for g in gathered]
        overlaps = sum(len(sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
        msg = (f'ddp-check epoch {trainer.current_epoch}: first-batch clips '
               f'{"disjoint" if overlaps == 0 else f"OVERLAP x{overlaps}"} across {trainer.world_size} ranks')
        if trainer.is_global_zero:
            LOG.info(msg)
        # sampler=balanced draws per-rank independent streams: rare chance collisions are expected and
        # harmless; any overlap is a real desync only under the partitioned DistributedSampler.
        if overlaps and str(pl_module.cfg.data.get('sampler', 'uniform')) != 'balanced':
            raise RuntimeError(msg)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.world_size < 2:
            return
        with torch.no_grad():
            flat = torch.cat([p.detach().flatten().double() for p in pl_module.model.parameters()])
            ck = torch.stack([flat.sum(), flat.abs().sum(), flat[::7919].sum()])
        gathered = pl_module.all_gather(ck)
        same = bool(torch.all(gathered == gathered[0]))
        maxdiff = float((gathered - gathered[0]).abs().max())
        msg = (f'ddp-check epoch {trainer.current_epoch}: params '
               f'{"identical" if same else f"DIFFER (max checksum diff {maxdiff:.3e})"} on {trainer.world_size} ranks')
        if trainer.is_global_zero:
            LOG.info(msg)
        if not same:
            raise RuntimeError(msg)


class FiniteLossGuard(L.Callback):
    """Finite-loss gate: stop immediately on a non-finite loss and keep the offending batch for inspection."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs
        if loss is None or bool(torch.isfinite(loss)):
            return
        rank = trainer.global_rank
        path = self.run_dir / f'nan_dump_rank{rank}_e{trainer.current_epoch}_b{batch_idx}.pt'

        def to_cpu(x):
            if torch.is_tensor(x):
                return x.detach().cpu()
            if isinstance(x, dict):
                return {k: to_cpu(v) for k, v in x.items()}
            return x

        torch.save({'losses': to_cpu(outputs.get('losses', {}) if isinstance(outputs, dict) else {}),
                    'batch': to_cpu(batch), 'epoch': trainer.current_epoch, 'batch_idx': batch_idx, 'rank': rank}, path)
        raise RuntimeError(f'non-finite loss at epoch {trainer.current_epoch} batch {batch_idx} (rank {rank}); batch dumped to {path}')
