"""Hydra + Lightning training entry point.

    python train.py experiment=example_codec                           # CPU smoke run on example_data/sample10
    torchrun --nproc_per_node=2 train.py experiment=codec              # stage 1 (configs/experiment/codec.yaml)
    torchrun --nproc_per_node=4 train.py experiment=prior diffusion.vae_ckpt=save/codec/best.ckpt   # stage 2
    python train.py ... resume=last                                    # continue from <run_dir>/last.ckpt
"""
import os
import sys
import time
import shutil
import logging
from pathlib import Path

import hydra
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from utils.misc import select_platform
from data.loco_dataset import read_meta
from network.checkpoint import load_weights
from network.lit_data import LocoDataModule
from network.factory import build_module
from network.callbacks import DDPSanityCallback, EMACallback, FiniteLossGuard, TrainMonitor


def finalize_cfg(cfg: DictConfig) -> DictConfig:
    """Derive the dependent values (clip_len, offset_frame, save_freq, ...), then freeze the config (struct mode)."""
    OmegaConf.set_struct(cfg, False)
    clip_len = int(cfg.data.past_frame) + int(cfg.data.future_frame)
    cfg.data.clip_len = clip_len
    if cfg.data.offset_frame is None:
        cfg.data.offset_frame = clip_len
    if cfg.trainer.save_freq is None:
        cfg.trainer.save_freq = int(cfg.trainer.epochs) // 10
    if cfg.trainer.best_after is None:
        cfg.trainer.best_after = int(cfg.trainer.epochs) // 10
    if cfg.resume == 'last':
        cfg.resume = os.path.join(cfg.paths.run_dir, 'last.ckpt')
    if cfg.continuity.enabled and float(cfg.model.cond_mask_prob) > 0:
        logging.getLogger('persona').warning('continuity.enabled=true forces model.cond_mask_prob=0 (100%% samples carry the prefix)')
    if cfg.data.get('skel_root_basis', 'joint') not in ('sole', 'joint'):    # what skel_offset[0] means; goes into the ckpt
        raise SystemExit(f"data.skel_root_basis must be 'sole' or 'joint', got {cfg.data.skel_root_basis!r}")
    if cfg.data.get('split', None):                                           # split_v2: pin the four files' sha256 into the run config
        from data.samplers import split_hashes
        cfg.data.split_sha = split_hashes(cfg.data.split)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg


def is_rank_zero_env():
    return int(os.environ.get('LOCAL_RANK', 0)) == 0 and int(os.environ.get('NODE_RANK', 0)) == 0


def prepare_run_dir(cfg):
    run_dir = Path(cfg.paths.run_dir)
    if run_dir.exists() and cfg.resume is None:
        if cfg.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise SystemExit(f'{run_dir} already exists: pass overwrite=true to replace it, or resume=last to continue')
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'config.yaml')


def setup_logging(run_dir, rank_zero):
    logger = logging.getLogger('persona')
    logger.disabled = False          # hydra's job_logging=disabled marks pre-existing loggers as disabled
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    if rank_zero:
        fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%d %b %Y %H:%M:%S')
        fh = logging.FileHandler(Path(run_dir) / 'log.txt', mode='a')
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def resolve_devices(cfg):
    devices = cfg.trainer.devices
    if 'LOCAL_WORLD_SIZE' in os.environ:            # torchrun
        devices = int(os.environ['LOCAL_WORLD_SIZE'])
    elif isinstance(devices, str) and devices.isdigit():
        devices = int(devices)
    accelerator = cfg.trainer.accelerator
    n = devices if isinstance(devices, int) else (torch.cuda.device_count() if accelerator in ('auto', 'gpu', 'cuda') else 1)
    strategy = 'ddp' if n > 1 else 'auto'
    return accelerator, devices, strategy


@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):
    start_time = time.time()
    cfg = finalize_cfg(cfg)
    rank_zero = is_rank_zero_env()
    if rank_zero:
        prepare_run_dir(cfg)
    run_dir = Path(cfg.paths.run_dir)
    log = setup_logging(run_dir, rank_zero)

    L.seed_everything(int(cfg.seed), workers=True)
    select_platform(32)                              # TF32 + cudnn.benchmark

    if rank_zero:
        log.info('Generative locomotion training with config: \n%s' % OmegaConf.to_yaml(cfg))

    meta = read_meta(cfg.data.path)
    if bool(cfg.diffusion.get('paired_data', False)):
        from data.paired_clips import PairedLocoDataModule
        datamodule = PairedLocoDataModule(cfg.data)
    else:
        datamodule = LocoDataModule(cfg.data)
    module = build_module(cfg, meta)
    if cfg.init_from:
        load_weights(module.model, cfg.init_from)
        log.info(f'Initialised weights from {cfg.init_from}')
    total_params = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    if rank_zero:
        log.info('Total parameters: %d' % total_params)
        log.info('\nModel structure: \n%s' % str(module.model))

    accelerator, devices, strategy = resolve_devices(cfg)
    ema = EMACallback(cfg.trainer.ema.decay) if cfg.trainer.ema.enabled else None
    callbacks = [c for c in (ema, TrainMonitor(run_dir, cfg, ema_callback=ema)) if c is not None]
    if cfg.checks.ddp_sanity:
        callbacks.append(DDPSanityCallback())
    if cfg.checks.finite_guard:
        callbacks.append(FiniteLossGuard(run_dir))

    trainer = L.Trainer(
        accelerator=accelerator, devices=devices, strategy=strategy, num_nodes=1,
        precision=cfg.trainer.precision, max_epochs=int(cfg.trainer.epochs),
        limit_train_batches=cfg.trainer.limit_train_batches,
        accumulate_grad_batches=int(cfg.trainer.get('accumulate', 1)),
        logger=TensorBoardLogger(save_dir=str(run_dir), name='', version='runtime', default_hp_metric=False),
        callbacks=callbacks, default_root_dir=str(run_dir),
        enable_checkpointing=False, use_distributed_sampler=str(cfg.data.get('sampler', 'uniform')) != 'balanced',
        num_sanity_val_steps=0, limit_val_batches=0, log_every_n_steps=1,
        enable_progress_bar=bool(cfg.trainer.progress_bar), enable_model_summary=False,
        profiler=cfg.trainer.profiler, deterministic=False, benchmark=None,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.resume)

    if trainer.is_global_zero:
        log.info('\nTotal training time: %.1f mins' % ((time.time() - start_time) / 60))


if __name__ == '__main__':
    main()
