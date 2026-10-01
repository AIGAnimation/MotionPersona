"""Loss terms.

``compute_losses`` holds the per-frame data / geometry / contact terms (switched by ``LossFlags``; the foot-contact
joints are looked up by name); ``seam_losses`` /
``seam_losses_xyz`` implement the history->future continuity terms (see configs/continuity/enabled.yaml).
"""
from dataclasses import dataclass

import torch

import utils.nn_transforms as nn_transforms


@dataclass(frozen=True)
class LossFlags:
    mse: bool = True
    geo3d: bool = True
    vel: bool = True
    accel: bool = False             # optional second-order term over the window INTERIOR
    accel_w: float = 5.0            # same weight as the first-order terms (loss_data_vel / loss_geo_xyz_vel)
    diff: bool = True
    contact: bool = False
    contact_w: float = 1.0           # weight of loss_foot_contact (at 1 it converges to ~7e-4, i.e. toothless)
    rot_req: str = '6d'
    past_recon: bool = True          # loss_past_recon on the first K predicted frames
    l_foot_idx: int = 59             # placeholder indices; use foot_indices_from_names()
    r_foot_idx: int = 55


def foot_indices_from_names(names, left='left_foot', right='right_foot'):
    """Foot joints for the contact loss, by name (the 59/55 defaults belong to a different skeleton)."""
    if left in names and right in names:
        return names.index(left), names.index(right)
    if 'LeftToeBase' in names and 'RightToeBase' in names:
        return names.index('LeftToeBase'), names.index('RightToeBase')
    raise ValueError('cannot find foot joints in skeleton names')


def masked_l2(a, b, mask):
    """Masked MSE loss, averaged over non-masked elements."""
    loss = (a - b) ** 2
    loss = (loss * mask).sum() / mask.sum()
    return loss


def flat_l2(a, b):
    """MSE loss flattened over all dims except batch."""
    return ((a - b) ** 2).mean(dim=list(range(1, a.ndim)))


def compute_losses(model_output, target, cond, flags, skel_parents, root_pos_mean, root_pos_std,
                   pred_future_motion, gt_future_motion, past_frame):
    """Standalone loss computation, independent of diffusion class.

    All tensors in [B, J, F, T] layout (T last). Returns a dict of loss terms including the weighted 'loss'.
    """
    batch_size = target.shape[0]

    fut_model_output = model_output[..., past_frame:]
    fut_target = target[..., past_frame:]
    fut_mask = cond['mask'].view(batch_size, 1, 1, -1)[..., past_frame:]
    loss_terms = {}

    if past_frame > 0 and flags.past_recon:
        pred_past = model_output[..., :past_frame]
        gt_past = cond['past_motion']
        loss_terms['loss_past_recon'] = 1.0 * ((pred_past - gt_past) ** 2).mean()

    if flags.mse:
        loss_terms['loss_data'] = 1 * masked_l2(fut_target, fut_model_output, fut_mask)

    if flags.vel:
        model_output_vel = fut_model_output[..., 1:] - fut_model_output[..., :-1]
        target_vel = fut_target[..., 1:] - fut_target[..., :-1]
        loss_terms['loss_data_vel'] = 5 * masked_l2(target_vel[:, :-1], model_output_vel[:, :-1], fut_mask[..., 1:])
    if flags.accel:
        # The default codec loss is first order everywhere except the seam (loss_seam_acc), so nothing penalises the
        # jerk spike a patch decoder puts at every token boundary. This is the same masked_l2 one derivative up,
        # over the whole future window.
        mo_acc = fut_model_output[..., 2:] - 2 * fut_model_output[..., 1:-1] + fut_model_output[..., :-2]
        tg_acc = fut_target[..., 2:] - 2 * fut_target[..., 1:-1] + fut_target[..., :-2]
        loss_terms['loss_data_acc'] = flags.accel_w * masked_l2(tg_acc[:, :-1], mo_acc[:, :-1], fut_mask[..., 2:])

    if flags.diff:
        loss_terms['loss_data_diff'] = 5 * flat_l2(cond['past_motion'][:, :, :, -1], pred_future_motion[:, :, :, 0]).mean()

    if flags.geo3d or flags.contact:
        target_rot = gt_future_motion.permute(0, 3, 1, 2)
        pred_rot = pred_future_motion.permute(0, 3, 1, 2)
        past_rot = cond['past_motion'].permute(0, 3, 1, 2)

        target_root_pos = target_rot[:, :, -2, :3] * root_pos_std + root_pos_mean
        pred_root_pos = pred_rot[:, :, -2, :3] * root_pos_std + root_pos_mean
        past_root_pos = past_rot[:, :, -2, :3] * root_pos_std + root_pos_mean

        skeletons = cond['skel_offset']
        parents = skel_parents[None]

        target_xyz = nn_transforms.neural_FK(target_rot[:, :, :-2], skeletons, target_root_pos, parents, rotation_type=flags.rot_req)
        pred_xyz = nn_transforms.neural_FK(pred_rot[:, :, :-2], skeletons, pred_root_pos, parents, rotation_type=flags.rot_req)

        if flags.geo3d:
            loss_terms['loss_geo_xyz'] = 0.5 * masked_l2(target_xyz.permute(0, 2, 3, 1), pred_xyz.permute(0, 2, 3, 1), fut_mask)

        if flags.vel:
            target_xyz_vel = target_xyz[:, 1:] - target_xyz[:, :-1]
            pred_xyz_vel = pred_xyz[:, 1:] - pred_xyz[:, :-1]
            loss_terms['loss_geo_xyz_vel'] = 5 * masked_l2(target_xyz_vel.permute(0, 2, 3, 1), pred_xyz_vel.permute(0, 2, 3, 1), fut_mask[..., 1:])
        if flags.accel:
            tgt_xyz_acc = target_xyz[:, 2:] - 2 * target_xyz[:, 1:-1] + target_xyz[:, :-2]
            prd_xyz_acc = pred_xyz[:, 2:] - 2 * pred_xyz[:, 1:-1] + pred_xyz[:, :-2]
            loss_terms['loss_geo_xyz_acc'] = flags.accel_w * masked_l2(tgt_xyz_acc.permute(0, 2, 3, 1), prd_xyz_acc.permute(0, 2, 3, 1), fut_mask[..., 2:])

        if flags.diff:
            past_xyz = nn_transforms.neural_FK(past_rot[:, :, :-2], skeletons, past_root_pos, parents, rotation_type=flags.rot_req)
            loss_terms['loss_geo_xyz_diff'] = 5 * flat_l2(past_xyz[:, -1], pred_xyz[:, 0]).mean()

        if flags.contact:
            l_foot_idx, r_foot_idx = flags.l_foot_idx, flags.r_foot_idx
            relevant_joints = [l_foot_idx, r_foot_idx]
            target_xyz_r = target_xyz.permute(0, 2, 3, 1)
            pred_xyz_r = pred_xyz.permute(0, 2, 3, 1)
            gt_joint_xyz = target_xyz_r[:, relevant_joints, :, :]
            gt_joint_vel = torch.linalg.norm(gt_joint_xyz[:, :, :, 1:] - gt_joint_xyz[:, :, :, :-1], axis=2)
            fc_mask = torch.unsqueeze((gt_joint_vel <= 0.01), dim=2).repeat(1, 1, 3, 1)
            pred_joint_xyz = pred_xyz_r[:, relevant_joints, :, :]
            pred_vel = pred_joint_xyz[:, :, :, 1:] - pred_joint_xyz[:, :, :, :-1]
            pred_vel[~fc_mask] = 0
            loss_terms['loss_foot_contact'] = flags.contact_w * masked_l2(pred_vel, torch.zeros_like(pred_vel), fut_mask[:, :, :, 1:])

    loss_terms['loss'] = sum(loss_terms.get(k, 0.) for k in [
        'loss_past_recon',
        'loss_data', 'loss_data_vel', 'loss_data_acc', 'loss_data_diff',
        'loss_geo_xyz', 'loss_geo_xyz_vel', 'loss_geo_xyz_acc', 'loss_geo_xyz_diff',
        'loss_foot_contact',
    ])

    return loss_terms


# ---------------------------------------------------------------------------------------------------------------
# history -> future seam losses
# ---------------------------------------------------------------------------------------------------------------
def _seam_mse(a, b, batch_dim=0, time_dim=-1):
    """Same normalisation as ``masked_l2`` with an all-ones [B,1,1,T] mask: sum over feature dims, mean over B and T."""
    sq = (a - b) ** 2
    dims = [d for d in range(sq.ndim) if d not in (batch_dim % sq.ndim, time_dim % sq.ndim)]
    return sq.sum(dim=dims).mean()


def _seam_terms(seq_pred, seq_gt, k, window, time_dim, drop_last_joint):
    """seq_* have the seam frame at index ``k`` along ``time_dim``; returns (pos, vel, acc) means.

    pos: frames [k, k+W); vel: first differences over frames [k-1, k+W) (W terms, the first one crosses the
    seam); acc: second differences over the same frames (W-1 terms).
    """
    def sl(x, a, b):
        idx = [slice(None)] * x.ndim
        idx[time_dim] = slice(a, b)
        return x[tuple(idx)]

    def diff(x):
        return sl(x, 1, None) - sl(x, 0, -1)

    if drop_last_joint:
        seq_pred_v, seq_gt_v = seq_pred[:, :-1], seq_gt[:, :-1]
    else:
        seq_pred_v, seq_gt_v = seq_pred, seq_gt
    pos = _seam_mse(sl(seq_pred, k, k + window), sl(seq_gt, k, k + window), time_dim=time_dim)
    d_pred, d_gt = diff(seq_pred_v), diff(seq_gt_v)                 # v_f = x_{f+1} - x_f
    vel = _seam_mse(sl(d_pred, k - 1, k - 1 + window), sl(d_gt, k - 1, k - 1 + window), time_dim=time_dim)
    if window < 2:                                                   # no acceleration term with a single frame
        return pos, vel, torch.zeros_like(pos)
    a_pred, a_gt = diff(d_pred), diff(d_gt)                          # a_f = x_{f+2} - 2 x_{f+1} + x_f
    acc = _seam_mse(sl(a_pred, k - 1, k - 1 + window - 1), sl(a_gt, k - 1, k - 1 + window - 1), time_dim=time_dim)
    return pos, vel, acc


def seam_losses(pred_full, gt_full, past_frame, window, w_pos, w_vel, w_acc):
    """Feature-space seam losses on the *pinned* sequence cat(gt[:K], pred[K:]); [B, J, F, T] layout.

    Velocity/acceleration terms drop the last pseudo-joint (foot contact), like ``loss_data_vel``.
    """
    k = past_frame
    seq_pred = torch.cat([gt_full[..., :k], pred_full[..., k:]], dim=-1)
    pos, vel, acc = _seam_terms(seq_pred, gt_full, k, window, time_dim=-1, drop_last_joint=True)
    return {'loss_seam_pos': w_pos * pos, 'loss_seam_vel': w_vel * vel, 'loss_seam_acc': w_acc * acc}


def seam_losses_xyz(pred_full, gt_full, cond, skel_parents, root_pos_mean, root_pos_std, rot_req,
                    past_frame, window, w_pos, w_vel, w_acc):
    """Same three terms on FK joint positions; FK is only run on frames [K-2, K+W) of the pinned sequence."""
    k = past_frame
    lo = k - 2
    seq_pred = torch.cat([gt_full[..., lo:k], pred_full[..., k:k + window]], dim=-1)   # (B,J,F,W+2)
    seq_gt = gt_full[..., lo:k + window]
    skeletons = cond['skel_offset']
    parents = skel_parents[None]

    def fk(seq):
        rot = seq.permute(0, 3, 1, 2)                                                  # (B,T',J,F)
        root = rot[:, :, -2, :3] * root_pos_std + root_pos_mean
        return nn_transforms.neural_FK(rot[:, :, :-2], skeletons, root, parents, rotation_type=rot_req)  # (B,T',J,3)

    xyz_pred, xyz_gt = fk(seq_pred), fk(seq_gt)
    pos, vel, acc = _seam_terms(xyz_pred, xyz_gt, 2, window, time_dim=1, drop_last_joint=False)
    return {'loss_seam_xyz_pos': w_pos * pos, 'loss_seam_xyz_vel': w_vel * vel, 'loss_seam_xyz_acc': w_acc * acc}
