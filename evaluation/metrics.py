"""The metrics library: every number of a rollout table from one rollout (paper Sec. 6.1, appendix "Contacts and
aggregation").

Inputs are the archive's `.motion.npz` (quats (T,J,4) wxyz, root (T,3) metres, contact (T,2) model flags, skel_offset
(J,3) metres, betas, leg) and `.traj.npz` (the body-scaled command sequence and the actual root path). References are
`save/eval/reference/<body>/<stem>.npz` rows (scripts/make_eval_refs.py).

Conventions:
  * ONE geometric contact detector for every method and the data: utils/foot_lock.detect_contacts on the toe joint
    (`left_foot` / `right_foot`), speed < 2 cm/frame and height < 5 cm above the sole plane y = 0, majority vote 3,
    min gap 2, min length 3. Skate / Pen. / IoU / contact ratio all use it.
  * Penetration is measured on the rigid foot model's sole corners (data/eval/feet_local.npz, four corners per foot
    in the ANKLE joint's local frame): the fraction of frames whose lowest corner is more than 1 cm below y = 0.
  * Jerk = the median third difference of the toe positions, cm/frame^3, over both feet's values (`toe_jerk_med`).
  * Seam = the FK joint displacement (cm per frame, mean over joints) at the block boundary frames b*hop-1 -> b*hop,
    reported next to the same statistic at every other frame (`seam_ref_cm`) and their ratio.
  * FPD: Gaussian fits (mean, covariance) on the rollout's frames and on the reference frames, Frechet distance
    |mu1-mu2|^2 + Tr(S1 + S2 - 2 (S1 S2)^1/2), in two per-frame spaces:
      pose     root-relative joint positions in the facing frame (root quaternion +Z forward), / leg  -> `fpd_pose`
      contact  per foot the lowest rigid-sole corner height above the floor and the horizontal toe speed, / leg,
               each dimension standardised by its std over the own-body reference-half takes  -> `fpd_contact`
    The contact space is the joint distribution of floor clearance and foot speed: it sees penetrating, floating and
    gliding feet, which no joint-position distribution can.
  * Reference takes drop their first frame (REF_DROP_HEAD): frame 0 of every capture is the rest pose at the origin
    (the root jumps ~0.6 m into frame 1), which would poison every velocity-based statistic of the take.
  * Spread descriptors (SPREAD_DYN / SPREAD_AMP) are computed in the pelvis-facing frame on the target body,
    leg-normalised, fps 30; foot clearance uses a leg-proportional ankle-height hysteresis contact of its own.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy import linalg
from scipy.ndimage import gaussian_filter1d

from data.bodies import REPO_NAMES, REPO_PARENTS
from utils import foot_lock as FL
from utils.nn_transforms import neural_FK

FPS = 30
FPD_VERSION = 1                      # bump when pose_features / contact_features / REF_DROP_HEAD change; reference/fpd_stats_v<version>
REF_DROP_HEAD = 1                    # leading rest frame(s) of a reference take
_J = REPO_NAMES.index
L_ANKLE, R_ANKLE = _J('left_ankle'), _J('right_ankle')
L_TOE, R_TOE = _J('left_foot'), _J('right_foot')
L_HIP, L_KNEE, R_HIP, R_KNEE = _J('left_hip'), _J('left_knee'), _J('right_hip'), _J('right_knee')
L_SHO, L_WRI, R_SHO, R_WRI = _J('left_shoulder'), _J('left_wrist'), _J('right_shoulder'), _J('right_wrist')
DETECTOR = dict(vel_thr=2.0, height_thr=5.0, window=3, min_gap=2, min_len=3)     # cm, cm, frames (foot_lock defaults)
SPREAD_DYN = ('joint_speed_leg', 'accel_leg', 'jerk_leg', 'hf_energy')
SPREAD_AMP = ('amplitude_leg', 'knee_range_deg', 'clearance_leg', 'bob_leg', 'arm_swing_deg')


# --------------------------------------------------------------------------------------------- geometry
def fk(quats, root, offsets, device='cpu'):
    """(T,J,4) wxyz, (T,3) m, (J,3) m -> (T,J,3) m world joint positions (neural_FK)."""
    q = torch.as_tensor(np.asarray(quats, np.float32), device=device)[None]
    r = torch.as_tensor(np.asarray(root, np.float32), device=device)[None]
    o = torch.as_tensor(np.asarray(offsets, np.float32), device=device)[None]
    xyz = neural_FK(q, o, r, torch.as_tensor(REPO_PARENTS, device=device)[None], rotation_type='q')[0]
    return xyz.cpu().numpy().astype(np.float64)


def leg_length(offsets):
    """Thigh + shank length (m) from the skeleton offsets: the unit of every `_leg` quantity."""
    o = np.asarray(offsets, np.float64)
    return float(np.linalg.norm(o[L_KNEE]) + np.linalg.norm(o[L_ANKLE]))


def sole_corner_heights(quats, root, offsets, feet_local):
    """feet_local (2,4,3) metres in the ankle joint's local frame (feet_local.npz `feet[body]`) -> (T,2,4) corner heights m."""
    pos, rots = FL.fk(np.asarray(quats, np.float64), np.asarray(root, np.float64), np.asarray(offsets, np.float64), REPO_PARENTS)
    out = np.zeros((len(pos), 2, 4))
    for s, ja in enumerate((L_ANKLE, R_ANKLE)):
        for k in range(4):
            out[:, s, k] = (pos[:, ja] + rots[ja].apply(np.tile(feet_local[s, k], (len(pos), 1))))[:, 1]
    return out


def geometric_contacts(xyz):
    """The one detector: toe joint, cm units, ground = the sole plane y = 0 -> (T,2) bool [left, right]."""
    out = np.zeros((len(xyz), 2), bool)
    for s, jt in enumerate((L_TOE, R_TOE)):
        toe = xyz[:, jt] * 100.0
        out[:, s] = FL.detect_contacts(toe, 0.0, **DETECTOR)[0]                    # (contacts, speed, height)
    return out


def skate_cm(xyz, contacts):
    """Mean horizontal toe speed on detected contact frames, cm/frame, averaged over the two feet (foot_lock.skate_metrics)."""
    vals = []
    for s, jt in enumerate((L_TOE, R_TOE)):
        sk = FL.skate_metrics(xyz[:, jt] * 100.0, contacts[:, s])
        if sk['mean'] is not None:
            vals.append(sk['mean'])
    return float(np.mean(vals)) if vals else np.nan


def penetration(corner_heights_m, thr_cm=1.0):
    """Fraction of frames whose lowest sole corner is more than thr_cm below the sole plane."""
    low = corner_heights_m.reshape(len(corner_heights_m), -1).min(1) * 100.0
    return float((low < -thr_cm).mean()), float(low.min())


def contact_iou(flag, det):
    a, b = np.asarray(flag, bool), np.asarray(det, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def seam(xyz, frame_of_block):
    """Joint displacement (cm/frame, mean over joints) at block boundaries vs everywhere else."""
    v = np.linalg.norm(np.diff(xyz, axis=0), axis=-1).mean(1) * 100.0      # v[t] = frame t -> t+1
    fb = np.asarray(frame_of_block)
    idx = fb[1:] - 1                                                        # the step INTO the first frame of block b
    idx = idx[(idx >= 0) & (idx < len(v))]
    mask = np.zeros(len(v), bool); mask[idx] = True
    return dict(seam_cm=float(v[mask].mean()), seam_max_cm=float(v[mask].max()), seam_ref_cm=float(v[~mask].mean()),
                seam_ratio=float(v[mask].mean() / max(v[~mask].mean(), 1e-9)))


def tracking(traj, leg):
    """Against the time-indexed command: horizontal root error (cm and legs), speed error (legs/s), facing error (deg)."""
    err = np.linalg.norm(np.asarray(traj['root_xz_m']) - np.asarray(traj['cmd_xz_m']), axis=-1)
    v_act = np.asarray(traj['root_speed_mps']); v_cmd = np.asarray(traj['cmd_speed_mps'])
    fe = np.abs(np.angle(np.exp(1j * (np.asarray(traj['root_facing_rad']) - np.asarray(traj['cmd_facing_rad'])))))
    return dict(track_cm=float(err.mean() * 100), track_leg=float(err.mean() / leg), track_max_cm=float(err.max() * 100),
                speed_err_leg=float(np.abs(v_act[1:] - v_cmd[1:]).mean() / leg), facing_err_deg=float(np.degrees(fe).mean()))


def smoothness(xyz):
    """Toe jerk: the third difference of the toe positions, cm/frame^3. The statistics are taken over BOTH feet's
    values, not over their per-frame mean: averaging the feet first mixes a planted foot (jerk ~ 0) with a swinging
    one at every frame, and the median would stop reporting planted-foot jitter. Jerk is population-sensitive (quiet
    styles read far lower than energetic ones), so only compare it across the same set of cases."""
    toe = xyz[:, [L_TOE, R_TOE]]
    per_foot = np.linalg.norm(np.diff(toe, n=3, axis=0), axis=-1) * 100.0            # (T-3, 2) cm/frame^3
    return dict(toe_jerk=float(per_foot.mean()), toe_jerk_med=float(np.median(per_foot)),
                toe_jerk_p95=float(np.percentile(per_foot, 95)),
                joint_jerk_med=float(np.median(np.linalg.norm(np.diff(xyz, n=3, axis=0), axis=-1) * 100.0)))


# --------------------------------------------------------------------------------------------- facing frame
def quat_rot(q, v):
    """Rotate vectors v (...,3) by wxyz quaternions q (...,4)."""
    w, qv = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def facing_frame(q0):
    """Root quaternions (T,4) wxyz -> unit forward (T,2) and left (T,2) xz vectors (+Z forward convention)."""
    f = quat_rot(np.asarray(q0, np.float64), np.array([0.0, 0.0, 1.0]))[:, [0, 2]]
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
    return f, np.stack([f[:, 1], -f[:, 0]], 1)


def to_facing(v, f, left):
    """World vectors (T,J,3) -> (T,J,3) [forward, left, up] in the per-frame facing frame."""
    fwd = v[..., 0] * f[:, None, 0] + v[..., 2] * f[:, None, 1]
    lat = v[..., 0] * left[:, None, 0] + v[..., 2] * left[:, None, 1]
    return np.stack([fwd, lat, v[..., 1]], -1)


# --------------------------------------------------------------------------------------------- FPD
def pose_features(xyz, root, q0, leg):
    """(T, J*3): root-relative joint positions in the facing frame (root quaternion +Z forward), / leg."""
    f, left = facing_frame(q0)
    return (to_facing(xyz - root[:, None], f, left) / leg).reshape(len(xyz), -1)


def _cdiff(x):
    v = np.empty_like(x)
    v[1:-1] = (x[2:] - x[:-2]) * (FPS / 2.0); v[0] = (x[1] - x[0]) * FPS; v[-1] = (x[-1] - x[-2]) * FPS
    return v


def contact_features(xyz, corner_heights_m, leg):
    """(T, 4): per foot the lowest rigid-sole corner height above the floor (the penetration measurement's geometry,
    sole_corner_heights) and the horizontal toe speed, both / leg (legs, legs/s). Planted feet sit at (0, 0),
    floating feet at (>0, 0), sinking feet at (<0, .), gliding feet at (~0, >0)."""
    sole = np.asarray(corner_heights_m, np.float64).min(2) / leg                            # (T, 2)
    vel_w = _cdiff(np.asarray(xyz, np.float64))
    toe_vxz = np.linalg.norm(vel_w[:, [L_TOE, R_TOE]][..., [0, 2]], axis=-1) / leg
    return np.concatenate([sole, toe_vxz], 1)


def gaussian_fit(feat):
    feat = np.asarray(feat, np.float64)
    return feat.mean(0), np.cov(feat, rowvar=False)


def sufficient_stats(feat):
    """(n, sum, sum of outer products) of a feature matrix; merge_stats() turns a list of them into one Gaussian."""
    feat = np.asarray(feat, np.float64)
    return len(feat), feat.sum(0), feat.T @ feat


def merge_stats(parts):
    """[(n, s1, s2)] -> (n, mu, cov) of all frames together (exact)."""
    n = sum(int(p[0]) for p in parts)
    s1 = sum(np.asarray(p[1], np.float64) for p in parts); s2 = sum(np.asarray(p[2], np.float64) for p in parts)
    mu = s1 / n
    return n, mu, (s2 - n * np.outer(mu, mu)) / max(n - 1, 1)


def standardise(mu, cov, scale):
    """Gaussian (mu, cov) expressed in units of `scale` (per-dimension std of the reference set)."""
    scale = np.asarray(scale, np.float64)
    return np.asarray(mu, np.float64) / scale, np.asarray(cov, np.float64) / np.outer(scale, scale)


def frechet(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        off = np.eye(len(mu1)) * eps
        covmean = linalg.sqrtm((s1 + off) @ (s2 + off))
    covmean = np.asarray(covmean).real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


def fpd(feat_gen, feat_ref, scale=None):
    """Frechet distance between the Gaussian fits of two feature matrices, optionally in units of `scale` per dimension."""
    g, r = gaussian_fit(feat_gen), gaussian_fit(feat_ref)
    if scale is not None:
        g, r = standardise(*g, scale), standardise(*r, scale)
    return frechet(*g, *r)


# --------------------------------------------------------------------------------------------- spread descriptors
def angle_deg(a, b):
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def p_range(x, lo=5, hi=95):
    return float(np.percentile(x, hi) - np.percentile(x, lo))


def height_contact(h_ankle, leg, lo=0.03, hi=0.08, min_flight=3):
    """0/1 contact from the ankle height with hysteresis (grounded below ground + lo*leg, airborne above ground +
    hi*leg, ground = the 5th percentile of the smoothed height), flights shorter than `min_flight` frames removed.
    Serves foot clearance only; every quality column uses geometric_contacts."""
    h = gaussian_filter1d(h_ankle, 1) - np.percentile(h_ankle, 5)
    lo, hi = lo * leg, hi * leg
    c = np.zeros(len(h), dtype=np.int64)
    state = 1 if h[0] < hi else 0
    for t in range(len(h)):
        if state == 1 and h[t] > hi:
            state = 0
        elif state == 0 and h[t] < lo:
            state = 1
        c[t] = state
    t = 0
    while t < len(c):
        if c[t] == 0:
            u = t
            while u < len(c) and c[u] == 0:
                u += 1
            if u - t < min_flight:
                c[t:u] = 1
            t = u
        else:
            t += 1
    return c


def spread_descriptors(xyz, root, q0, leg):
    """The descriptors behind the spread-kept columns (SPREAD_DYN + SPREAD_AMP), pelvis-facing frame, leg units.

    dynamics: joint speed / acceleration / jerk (root-relative, facing frame; leg/s, leg/s^2, leg/s^3) and the share
    of velocity power above 3 Hz; amplitude: root-relative joint-position spread, knee flexion range (5-95th pct),
    foot clearance (ankle height in swing minus in stance), pelvis bob (root height minus its trend) and arm swing
    (sagittal shoulder-wrist angle range)."""
    T = len(xyz)
    f, left = facing_frame(q0)
    loc = to_facing(xyz - root[:, None], f, left)                         # (T,J,3) in the facing frame
    cl, cr = height_contact(xyz[:, L_ANKLE, 1], leg), height_contact(xyz[:, R_ANKLE, 1], leg)
    clear = []
    for a, c in ((L_ANKLE, cl), (R_ANKLE, cr)):
        h = xyz[:, a, 1]
        if (c == 1).sum() > 5 and (c == 0).sum() > 5:
            clear.append(np.percentile(h[c == 0], 95) - np.percentile(h[c == 1], 5))
    kl = angle_deg(xyz[:, L_HIP] - xyz[:, L_KNEE], xyz[:, L_ANKLE] - xyz[:, L_KNEE])
    kr = angle_deg(xyz[:, R_HIP] - xyz[:, R_KNEE], xyz[:, R_ANKLE] - xyz[:, R_KNEE])

    def sagittal(a, b):                                                   # angle of (b - a) from straight down, + = forward
        d = loc[:, b] - loc[:, a]
        return np.degrees(np.arctan2(d[:, 0], -d[:, 2]))
    ybob = float((root[:, 1] - gaussian_filter1d(root[:, 1], 15)).std())
    d1 = np.diff(loc, axis=0) * FPS
    d2 = np.diff(loc, n=2, axis=0) * FPS ** 2
    d3 = np.diff(loc, n=3, axis=0) * FPS ** 3
    if T > 8:
        spec = np.abs(np.fft.rfft(d1 - d1.mean(0, keepdims=True), axis=0)) ** 2
        freqs = np.fft.rfftfreq(d1.shape[0], d=1.0 / FPS)
        hf_energy = float(spec[freqs > 3.0].sum() / (spec.sum() + 1e-12))
    else:
        hf_energy = np.nan
    amplitude = float(loc.reshape(T, -1).std(0).mean()) if T > 1 else np.nan
    return dict(
        joint_speed_leg=float(np.linalg.norm(d1, axis=-1).mean()) / leg,
        accel_leg=float(np.linalg.norm(d2, axis=-1).mean()) / leg,
        jerk_leg=float(np.linalg.norm(d3, axis=-1).mean()) / leg,
        hf_energy=hf_energy,
        amplitude_leg=amplitude / leg,
        knee_range_deg=float(np.mean([p_range(kl), p_range(kr)])),
        clearance_leg=float(np.mean(clear)) / leg if clear else np.nan,
        bob_leg=ybob / leg,
        arm_swing_deg=float(np.mean([p_range(sagittal(L_SHO, L_WRI)), p_range(sagittal(R_SHO, R_WRI))])),
    )


# --------------------------------------------------------------------------------------------- one rollout
@dataclass
class Rollout:
    quats: np.ndarray
    root: np.ndarray
    contact: np.ndarray
    offsets: np.ndarray
    leg: float
    body: int
    traj: dict | None = None
    frame_of_block: np.ndarray | None = None

    @classmethod
    def load(cls, motion_npz, traj_npz=None):
        z = np.load(motion_npz)
        t = dict(np.load(traj_npz)) if traj_npz is not None else None
        return cls(quats=z['quats'].astype(np.float64), root=z['root'].astype(np.float64), contact=z['contact'], offsets=z['skel_offset'].astype(np.float64),
                   leg=float(z['leg']), body=int(z['body']), traj=t, frame_of_block=None if t is None else np.asarray(t['frame_of_block']))


def rollout_metrics(r: Rollout, feet_local=None, device='cpu', with_descriptors=True):
    """All per-rollout numbers as one flat dict + the FPD feature matrices {'pose': (T, J*3), 'contact': (T, 4)} (the
    last only with `feet_local` = feet_local.npz feet[body] (2,4,3), which also enables penetration)."""
    xyz = fk(r.quats, r.root, r.offsets, device)
    det = geometric_contacts(xyz)
    flag = np.asarray(r.contact) > 0.5
    out = dict(frames=len(xyz), skate=skate_cm(xyz, det), iou=contact_iou(flag, det), contact_ratio=float(det.mean()),
               flag_ratio=float(flag.mean()))
    toe_h = np.stack([xyz[:, L_TOE, 1], xyz[:, R_TOE, 1]], 1) * 100.0
    out['h_contact_cm'] = float(toe_h[det].mean()) if det.any() else np.nan
    out['minh_toe_cm'] = float(toe_h.min())
    out.update(smoothness(xyz))
    ch = None
    if feet_local is not None:
        ch = sole_corner_heights(r.quats, r.root, r.offsets, feet_local)
        out['pen'], out['min_sole_cm'] = penetration(ch)
    if r.frame_of_block is not None:
        out.update(seam(xyz, r.frame_of_block))
    if r.traj is not None:
        out.update(tracking(r.traj, r.leg))
    feats = {'pose': pose_features(xyz, r.root, r.quats[:, 0], r.leg)}
    if ch is not None:
        feats['contact'] = contact_features(xyz, ch, r.leg)
    if with_descriptors:
        leg = leg_length(r.offsets)
        assert abs(leg - r.leg) < 1e-3, (leg, r.leg)
        out.update(spread_descriptors(xyz, r.root, r.quats[:, 0], leg))
    return out, feats


def reference_metrics(z, feet_local=None, device='cpu', drop_head=REF_DROP_HEAD):
    """The data row from a reference take: quality metrics with the captured labels as the flag. The take's leading
    rest frame(s) are dropped (REF_DROP_HEAD)."""
    off = z['skel_offset'].astype(np.float64)
    r = Rollout(quats=z['rotations'][drop_head:].astype(np.float64), root=z['root_pos'][drop_head:].astype(np.float64),
                contact=np.asarray(z['foot_contact'])[drop_head:], offsets=off, leg=leg_length(off), body=int(z['body']))
    return rollout_metrics(r, feet_local=feet_local, device=device)


# --------------------------------------------------------------------------------------------- table-level
def spread_kept(gen_by_key, ref_by_key):
    """gen_by_key / ref_by_key: {persona: value} within one style -> std(gen) / std(ref) over the shared personas."""
    ks = sorted(set(gen_by_key) & set(ref_by_key))
    if len(ks) < 3:
        return np.nan
    g = np.array([gen_by_key[k] for k in ks]); r = np.array([ref_by_key[k] for k in ks])
    return float(np.nanstd(g) / (np.nanstd(r) + 1e-12))
