"""Canonical history: every rollout on a body starts from the same 45-frame window of the take `p02_neutral_fw`
retargeted to that body (the take's rows exist in the retargeted dataset for all 128 bodies), so the initialisation
cannot reveal the persona or the style. The histories are shipped in data/eval/canonical/<body:03d>.npz.

Selection: among windows [e - F, e) ending at a LEFT-foot contact onset e (the stored labels, which are what the codec
encodes), inside the middle of the take, pick the one whose mean pelvis speed is closest to the route's v0 (legs/s)
and, at equal closeness, the steadiest. The window is then translated so its last frame's root xz is the route origin
and rotated about +Y so its last frame's facing (hip forward, the data's definition) is the route's initial facing
(+Z). Root height is untouched (y = 0 is the sole plane on every body).
"""
from dataclasses import dataclass

import numpy as np

from evaluation.route import yaw_of
from realtime.model_loop import hip_forward

FPS = 30


def quat_mul(a, b):
    """wxyz Hamilton product, broadcasting."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def rotate_about_y(rotations, root_pos, angle):
    """Rotate a clip about +Y by `angle` (rad) around the origin: root quats pre-multiplied, root xz rotated."""
    q = np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0])
    rot = np.array(rotations, np.float64).copy()
    rot[:, 0] = quat_mul(q[None], rot[:, 0])
    c, s = np.cos(angle), np.sin(angle)
    pos = np.array(root_pos, np.float64).copy()
    x, z = pos[:, 0].copy(), pos[:, 2].copy()
    pos[:, 0], pos[:, 2] = c * x + s * z, -s * x + c * z          # same map as (sin, cos) yaw on +Z
    return rot, pos


@dataclass
class CanonicalWindow:
    rotations: np.ndarray   # (F, J, 4) wxyz
    root_pos: np.ndarray    # (F, 3) m, last frame xz = 0
    foot_contact: np.ndarray  # (F, 2)
    start: int
    end: int
    speed_leg: float
    yaw_applied: float


def select_window(rotations, root_pos, foot_contact, offsets, leg, v0_leg, *, F=45, fps=FPS, margin=0.15):
    """-> (start, end, speed_leg) of the chosen window; windows end at a left-foot contact onset."""
    T = len(root_pos)
    lo, hi = int(margin * T), int((1 - margin) * T)
    onsets = np.flatnonzero(np.diff(foot_contact[:, 0].astype(np.int64)) == 1) + 1
    best = None
    for e in onsets:
        s = e - F
        if s < lo or e > hi:
            continue
        xz = root_pos[s:e, [0, 2]]
        v = np.linalg.norm(np.diff(xz, axis=0), axis=-1) * fps
        speed = float(v.mean() / leg)
        score = (abs(speed - v0_leg), float(v.std() / leg))
        if best is None or score < best[0]:
            best = (score, s, e, speed)
    if best is None:
        raise ValueError('no left-foot contact onset with a full window inside the take')
    _, s, e, speed = best
    return int(s), int(e), speed


def canonical_window(rotations, root_pos, foot_contact, offsets, leg, v0_leg, *, F=45, fps=FPS, target_yaw=0.0, window=None):
    """`window=(start, end)` reuses frames chosen elsewhere (the own-body row), else selects here."""
    if window is None:
        s, e, speed = select_window(rotations, root_pos, foot_contact, offsets, leg, v0_leg, F=F, fps=fps)
    else:
        s, e = int(window[0]), int(window[1])
        v = np.linalg.norm(np.diff(np.asarray(root_pos[s:e], np.float64)[:, [0, 2]], axis=0), axis=-1) * fps
        speed = float(v.mean() / leg)
    rot = np.array(rotations[s:e], np.float64); pos = np.array(root_pos[s:e], np.float64); fc = np.array(foot_contact[s:e], np.float32)
    pos[:, [0, 2]] -= pos[-1, [0, 2]]                                # last frame at the route origin
    fwd = hip_forward(rot[-1, 0], offsets)
    angle = target_yaw - float(yaw_of(fwd[[0, 2]]))
    rot, pos = rotate_about_y(rot, pos, angle)
    fwd2 = hip_forward(rot[-1, 0], offsets)
    assert abs(np.angle(np.exp(1j * (yaw_of(fwd2[[0, 2]]) - target_yaw)))) < 1e-6
    return CanonicalWindow(rotations=rot.astype(np.float32), root_pos=pos.astype(np.float32), foot_contact=fc,
                           start=s, end=e, speed_leg=speed, yaw_applied=angle)
