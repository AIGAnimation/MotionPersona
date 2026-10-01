"""Thin Lightning adapter around network/diffusion.py + network/losses.py."""
import os

import numpy as np
import torch
import lightning as L
from omegaconf import DictConfig, OmegaConf

import utils.nn_transforms as nn_transforms
from data.augment import featurize_batch
from network.checkpoint import build_model, build_scheduler, public_meta
from network.diffusion import ContinuityCfg, ddpm_sample, de_delta, diffusion_forward
from utils.foot_lock import FootLockConfig, foot_lock_motion
from network.losses import LossFlags, compute_losses, foot_indices_from_names, seam_losses, seam_losses_xyz


class MotionDiffusionModule(L.LightningModule):
    def __init__(self, cfg, meta):
        super().__init__()
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg
        pub = public_meta(meta)
        self.save_hyperparameters({'cfg': OmegaConf.to_container(cfg, resolve=True), 'meta': pub})
        self.T_pose = meta.get('T_pose')                       # Motion template for BVH export (not saved)
        self.names = pub['names']

        self.model = build_model(cfg, meta)
        self.scheduler = build_scheduler(cfg.diffusion)
        self.past_frame = int(cfg.data.past_frame)
        self.clip_len = int(cfg.data.past_frame) + int(cfg.data.future_frame)
        self.delta = bool(cfg.diffusion.delta)
        self.cond_mask_prob = float(self.model.cond_mask_prob)
        self.hist_guidance = 1.0                               # see `guided_by_history` (1 = plain conditional)
        c = cfg.continuity
        self.continuity = ContinuityCfg(bool(c.enabled), int(c.seam.window), float(c.seam.w_pos),
                                        float(c.seam.w_vel), float(c.seam.w_acc), bool(c.seam.use_xyz))
        l_foot, r_foot = foot_indices_from_names(self.names) if cfg.loss.contact else (59, 55)
        self.loss_flags = LossFlags(mse=bool(cfg.loss.mse), geo3d=bool(cfg.loss.geo3d), vel=bool(cfg.loss.vel),
                                    diff=bool(cfg.loss.diff), contact=bool(cfg.loss.contact), contact_w=float(cfg.loss.get('contact_w', 1.0)), rot_req=cfg.model.rot_req,
                                    accel=bool(cfg.loss.get('accel', False)), accel_w=float(cfg.loss.get('accel_w', 5.0)),
                                    past_recon=not self.continuity.enabled, l_foot_idx=l_foot, r_foot_idx=r_foot)

        self.register_buffer('root_pos_mean', torch.tensor(pub['root_pos_mean'], dtype=torch.float32), persistent=False)
        self.register_buffer('root_pos_std', torch.tensor(pub['root_pos_std'], dtype=torch.float32), persistent=False)
        self.register_buffer('skel_parents', torch.tensor(pub['parents'], dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------ data
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if 'rotations' in batch:                               # raw clips from LocoDataset -> features (GPU)
            with torch.no_grad():
                # end_frames > 0 only for the in-betweening switch (network/latfm.py, data.end_token_prob); the
                # default 0 leaves the batch unchanged.
                return featurize_batch(batch, self.root_pos_mean, self.root_pos_std, self.past_frame,
                                       end_frames=int(getattr(self, 'end_frames', 0)))
        return batch

    def featurize(self, raw_batch):
        """Public helper for callbacks/tests: raw batch (any device) -> featurized batch on this module's device."""
        raw_batch = self.transfer_batch_to_device(raw_batch, self.device, 0)
        return self.on_after_batch_transfer(raw_batch, 0)

    # ------------------------------------------------------------------ train
    def _losses(self, out, cond):
        K = self.past_frame
        losses = compute_losses(out['model_output'], out['x0'], cond, self.loss_flags, self.skel_parents,
                                self.root_pos_mean, self.root_pos_std, out['pred_full'][..., K:], out['gt_full'][..., K:], K)
        if self.continuity.enabled:
            c = self.continuity
            seam = seam_losses(out['pred_full'], out['gt_full'], K, c.window, c.w_pos, c.w_vel, c.w_acc)
            if c.use_xyz and self.loss_flags.geo3d:
                seam.update(seam_losses_xyz(out['pred_full'], out['gt_full'], cond, self.skel_parents, self.root_pos_mean,
                                            self.root_pos_std, self.loss_flags.rot_req, K, c.window, c.w_pos, c.w_vel, c.w_acc))
            losses['loss'] = losses['loss'] + sum(seam.values())
            losses.update(seam)
        return losses

    def forward_diffuse(self, batch, **kw):
        cond = batch['conditions']
        x_start = batch['data_delta'] if self.delta else batch['data']
        return diffusion_forward(self.model, self.scheduler, x_start, cond, past_frame=self.past_frame, delta=self.delta,
                                 cond_mask_prob=self.cond_mask_prob, continuity_enabled=self.continuity.enabled, **kw)

    def training_step(self, batch, batch_idx):
        out = self.forward_diffuse(batch)
        losses = self._losses(out, batch['conditions'])
        for k, v in losses.items():
            self.log(f'train/{k}', v.detach(), on_step=False, on_epoch=True, sync_dist=True, batch_size=1,
                     logger=False, prog_bar=(k == 'loss'))
        return {'loss': losses['loss'], 'losses': {k: v.detach() for k, v in losses.items()}}

    def configure_optimizers(self):
        tr = self.cfg.trainer
        opt = torch.optim.AdamW(self.model.parameters(), lr=float(tr.lr), weight_decay=float(tr.weight_decay))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tr.epochs), eta_min=float(tr.lr) * float(tr.lr_min_ratio))
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'interval': 'epoch', 'frequency': 1}}

    # ------------------------------------------------------------------ eval / export
    @torch.no_grad()
    def x0_at_random_t(self, batch):
        """One denoising pass at a random t. Returns (gt_full, pred_full) in [B,T,J,F]."""
        out = self.forward_diffuse(batch)
        return out['gt_full'].permute(0, 3, 1, 2), out['pred_full'].permute(0, 3, 1, 2)

    def guided_by_history(self, call, history):
        """Classifier-free guidance on the *history*, which is the only condition training ever drops
        (`cond_mask_prob`, 0.15 in the released prior): the trained null branch is "no past pose, just follow the
        trajectory and the prompt". With w = `self.hist_guidance`,

            w = 1   the plain conditional prediction (no extra cost, the default everywhere)
            w < 1   loosens the past pose's grip, so a freshly switched prompt is not outvoted by the history of
                    the previous one (what the realtime demo does for a few chunks after a prompt change)
            w > 1   tightens it

        `call(history)` runs the network; a second forward is only paid when w != 1."""
        out = call(history)
        w = float(self.hist_guidance)
        if w == 1.0:
            return out
        free = call(torch.zeros_like(history))
        return free + w * (out - free)

    @torch.no_grad()
    def sample(self, batch, generator=None):
        """Full DDPM sampling loop (prefix pinned every step under continuity). Returns (gt_full, sample) in [B,T,J,F]."""
        cond = batch['conditions']
        x_start = batch['data_delta'] if self.delta else batch['data']
        x = ddpm_sample(self.model, self.scheduler, cond, past_frame=self.past_frame, clip_len=self.clip_len, delta=self.delta,
                        continuity_enabled=self.continuity.enabled,
                        num_inference_steps=self.cfg.diffusion.num_inference_steps, generator=generator,
                        guide=self.guided_by_history if self.hist_guidance != 1.0 else None)
        gt_full = de_delta(x_start.permute(0, 2, 3, 1), cond['past_motion'], self.delta)
        return gt_full.permute(0, 3, 1, 2), de_delta(x, cond['past_motion'], self.delta).permute(0, 3, 1, 2)

    def export_bvh(self, motion_feat, skeletons, save_path, prefix, foot_lock=None):
        """Sample export: motion_feat [B,T,J+2,F] (6d) -> motion_i.<prefix>.bvh.

        ``foot_lock`` (a ``FootLockConfig`` / the ``export.foot_lock`` config node) with ``enabled=true`` applies
        ``utils.foot_lock`` to the written clip (toe contact lock + leg IK, in the BVH's own units); with
        ``keep_unlocked`` the untouched clip also goes to motion_i.<prefix>_nolock.bvh. Returns the per-sample
        foot-lock info dicts ([] when disabled).
        """
        os.makedirs(save_path, exist_ok=True)
        rotations = nn_transforms.repr6d2quat(motion_feat[:, :, :-2]).cpu().numpy()
        root_poses = motion_feat[:, :, -2, :3].cpu().numpy()
        skeletons = skeletons.cpu().numpy()
        rp_mean, rp_std = self.root_pos_mean.cpu().numpy(), self.root_pos_std.cpu().numpy()
        fl_cfg = FootLockConfig.from_cfg(foot_lock) if foot_lock is not None else None
        infos = []
        for i in range(motion_feat.shape[0]):
            tpl = self.T_pose.copy()
            tpl.offsets = skeletons[i]
            tpl.rotations = rotations[i]
            tpl.positions = np.zeros((rotations[i].shape[0], tpl.positions.shape[1], tpl.positions.shape[2]))
            tpl.positions[:, 0] = root_poses[i] * rp_std + rp_mean
            if fl_cfg is not None and fl_cfg.enabled:
                if fl_cfg.keep_unlocked:
                    tpl.copy().export(os.path.join(save_path, f'motion_{i}.{prefix}_nolock.bvh'), save_ori_scal=True)
                tpl, info = foot_lock_motion(tpl, fl_cfg, pin=self.past_frame)
                infos.append(info)
            tpl.export(os.path.join(save_path, f'motion_{i}.{prefix}.bvh'), save_ori_scal=True)
        return infos
