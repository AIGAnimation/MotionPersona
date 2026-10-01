"""Foot locking for locomotion clips: toe-contact locking + two-bone leg IK, after Daniel Holden's
"Inverse Kinematics and Foot Locking" (https://theorangeduck.com/page/inverse-kinematics-foot-locking).

Entry points
    FootLockConfig                      every parameter, defaults = configs/config.yaml ``export.foot_lock``
    foot_lock(rotations, root_pos, offsets, parents, names, frametime, cfg, pin) -> FootLockResult
    foot_lock_motion(motion, cfg, pin)  same on a ``utils.bvh_motion.Motion`` (handles its scaling_factor)
    measure(...)                        contact detection + skating metrics only (no modification)
    summarize(infos)                    JSON-able aggregate of several ``FootLockResult.info`` dicts

Pipeline per clip
    1. FK -> toe / ankle / knee / hip world positions
    2. contact detection on the toe: speed < vel_thr AND height above ground < height_thr, majority vote,
       gaps <= min_gap filled, runs < min_len dropped
    3. per-foot toe target
         runtime : the article's FootLockingState machine (lock on contact, unlock when the contact ends or the
                   input drifts > unlock_dist away) with cubic inertialization of the transitions
         offline : position-based constraint solve over the whole clip (hard "same toe position on consecutive
                   contact frames", soft "keep the source frame-to-frame toe / pelvis offsets and hip-toe length"),
                   the first ``pin`` (history) frames held fixed
    4. two-bone leg IK on hip + knee about the knee's own hinge axis so the ankle reaches
       target_toe + (ankle - toe), soft extension clamp (never below the input pose's own reach), then an ankle
       look-at so the toe lands exactly on the target
Units are those of the skeleton; the defaults assume centimetres and 30 fps. Pure numpy/scipy.
"""
import dataclasses
import math
from dataclasses import dataclass, field, fields

import numpy as np
from scipy.spatial.transform import Rotation as R

LEG_CHAINS = (
    {'left': ('left_hip', 'left_knee', 'left_ankle', 'left_foot'), 'right': ('right_hip', 'right_knee', 'right_ankle', 'right_foot')},
    {'left': ('LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase'), 'right': ('RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase')},
)


@dataclass
class FootLockConfig:
    enabled: bool = False
    mode: str = 'runtime'          # runtime | offline
    keep_unlocked: bool = True     # export: also write motion_i.<prefix>_nolock.bvh
    # contact detection (cm, cm/frame)
    vel_thr: float = 2.0           # toe speed below this (2.0 cm/frame = 0.6 m/s @ 30 fps)
    height_thr: float = 5.0        # toe height above ground below this
    vote: int = 3                  # majority-vote window (frames)
    min_gap: int = 2               # fill non-contact gaps of <= this many frames
    min_len: int = 3               # drop contact runs shorter than this
    ground: float = None           # ground height; None = lowest toe of the clip
    plant: bool = False            # pull locked toes down to the ground height
    # runtime locking (article part 2)
    blend: float = 0.1             # inertialization blend time (s)
    lock_dist: float = 15.0
    unlock_dist: float = 25.0
    # offline solve (article part 4)
    soft: float = 0.05
    hard: float = 0.9
    iters: int = 5000
    pin: int = None                # frames held fixed; None = the caller's history length
    # IK (article part 1)
    max_ext: float = 0.98          # max hip-heel reach as a fraction of thigh + shin
    softening: float = 0.5         # extension clamp softening (cm)

    @classmethod
    def from_cfg(cls, node=None, **overrides):
        """Build from None / dict / DictConfig / FootLockConfig (+ keyword overrides); unknown keys raise."""
        if isinstance(node, cls):
            base = dataclasses.asdict(node)
        elif node is None:
            base = {}
        else:
            base = {str(k): node[k] for k in node.keys()}
        base.update(overrides)
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(base) - names)
        if unknown:
            raise ValueError(f'unknown foot_lock keys: {unknown}')
        cfg = cls(**base)
        if cfg.mode not in ('runtime', 'offline'):
            raise ValueError(f"foot_lock.mode must be 'runtime' or 'offline', got {cfg.mode!r}")
        return cfg

    def to_dict(self):
        return dataclasses.asdict(self)


@dataclass
class FootLockResult:
    rotations: np.ndarray          # (T, J, 4) wxyz, float64
    root_pos: np.ndarray           # (T, 3)
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- kinematics
def resolve_chains(names):
    names = list(names)
    for chains in LEG_CHAINS:
        if all(n in names for c in chains.values() for n in c):
            return {k: tuple(names.index(n) for n in c) for k, c in chains.items()}
    raise ValueError('cannot find the leg joints (hip/knee/ankle/toe) in the skeleton names')


def fk(rot_wxyz, root_pos, offsets, parents):
    """Global joint positions (T,J,3) and orientations (list of Rotation, one per joint, batched over T)."""
    T, J = rot_wxyz.shape[:2]
    pos = np.zeros((T, J, 3))
    glob = [None] * J
    for j, p in enumerate(parents):
        local = R.from_quat(rot_wxyz[:, j][:, [1, 2, 3, 0]])
        if p < 0:
            glob[j] = local
            pos[:, j] = root_pos
        else:
            glob[j] = glob[p] * local
            # tile, not broadcast_to: scipy 1.17's Rotation.apply rejects a read-only buffer (and for T = 1 a
            # broadcast view is already "contiguous", so ascontiguousarray would not copy it either)
            pos[:, j] = pos[:, p] + glob[p].apply(np.tile(offsets[j], (T, 1)))
    return pos, glob


def rot_between(p, q):
    """Shortest rotation taking direction p to direction q (QuaternionBetween in the article)."""
    c = np.cross(p, q)
    w = math.sqrt(float(np.dot(p, p) * np.dot(q, q))) + float(np.dot(p, q))
    quat = np.array([c[0], c[1], c[2], w])
    n = np.linalg.norm(quat)
    if n < 1e-8:  # anti-parallel: 180 deg about any axis perpendicular to p
        axis = np.cross(p, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(p, [0.0, 1.0, 0.0])
        return R.from_rotvec(axis / np.linalg.norm(axis) * math.pi)
    return R.from_quat(quat / n)


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-12)


def two_bone_ik(a, b, c, target, R_parent, R_hip, R_knee, side, max_ext, softening):
    """Article part 1: new local hip / knee rotations bringing the heel c to `target` (a = hip, b = knee).

    Rotation axis = the knee's current hinge axis (falls back to the article's side-vector construction when the
    leg is straight); r0/r1 change the hip/knee interior angles by the cosine-rule deltas, r2 swings the whole
    leg onto the target direction. Returns (hip_local, knee_local, clamped_target, clamped_flag)."""
    lab = np.linalg.norm(b - a)
    lcb = np.linalg.norm(c - b)
    d = target - a
    lat = np.linalg.norm(d)
    clamped = False
    if lat > max_ext - softening:
        sat = 1.0 - math.exp(-max(lat - max_ext + softening, 0.0) / softening)
        target = a + d * ((max_ext - softening + softening * sat) / lat)
        lat = np.linalg.norm(target - a)
        clamped = True

    hinge = np.cross(c - b, b - a)
    if np.linalg.norm(hinge) > 1e-3 * lab * lcb:
        axis = _unit(hinge)
    else:
        axis_dwn = _unit(c - a)
        axis_fwd = _unit(np.cross(axis_dwn, side))
        axis = _unit(np.cross(axis_dwn, axis_fwd))

    acab0 = math.acos(np.clip(np.dot(_unit(c - a), _unit(b - a)), -1.0, 1.0))
    babc0 = math.acos(np.clip(np.dot(_unit(a - b), _unit(c - b)), -1.0, 1.0))
    acab1 = math.acos(np.clip((lab * lab + lat * lat - lcb * lcb) / (2.0 * lab * lat), -1.0, 1.0))
    babc1 = math.acos(np.clip((lab * lab + lcb * lcb - lat * lat) / (2.0 * lab * lcb), -1.0, 1.0))

    r0 = R.from_rotvec(axis * (acab1 - acab0))
    r1 = R.from_rotvec(axis * (babc1 - babc0))
    r2 = rot_between(c - a, target - a)
    hip_local = R_parent.inv() * r2 * r0 * R_hip
    knee_local = R_hip.inv() * r1 * R_knee
    return hip_local, knee_local, target, clamped


def solve_leg(rot, pos, orient, offsets, parents, chain, toe_targets, max_ext_ratio, softening, eps=1e-4):
    """IK + ankle look-at on one leg for every frame whose toe target differs from the FK toe.
    `rot` (T,J,4 wxyz) is modified in place; `pos`/`orient` are the FK of the input. Returns per-leg stats."""
    hip, knee, ankle, toe = chain
    parent = parents[hip]
    T = rot.shape[0]
    n_ik = n_clamp = 0
    toe_err = np.zeros(T)
    knee_delta = np.zeros(T)
    for i in range(T):
        tt = toe_targets[i]
        if np.linalg.norm(tt - pos[i, toe]) < eps:
            continue
        a, b, c = pos[i, hip], pos[i, knee], pos[i, ankle]
        R_par, R_hip, R_knee, R_ankle = orient[parent][i], orient[hip][i], orient[knee][i], orient[ankle][i]
        side = R_par.apply([1.0, 0.0, 0.0])
        heel_target = tt + (c - pos[i, toe])
        reach = np.linalg.norm(offsets[knee]) + np.linalg.norm(offsets[ankle])
        max_ext = max(max_ext_ratio * reach, np.linalg.norm(c - a))     # never below the input's own reach
        hip_l, knee_l, _, clamped = two_bone_ik(a, b, c, heel_target, R_par, R_hip, R_knee, side, max_ext, softening)
        n_ik += 1
        n_clamp += int(clamped)
        R_hip2 = R_par * hip_l
        R_knee2 = R_hip2 * knee_l
        b2 = a + R_hip2.apply(offsets[knee])
        c2 = b2 + R_knee2.apply(offsets[ankle])
        ankle_l = R_knee.inv() * R_ankle
        R_ankle2 = R_knee2 * ankle_l
        toe2 = c2 + R_ankle2.apply(offsets[toe])
        fix = rot_between(toe2 - c2, tt - c2)                            # ankle look-at
        R_ankle3 = fix * R_ankle2
        ankle_l3 = R_knee2.inv() * R_ankle3
        toe3 = c2 + R_ankle3.apply(offsets[toe])
        toe_err[i] = np.linalg.norm(toe3 - tt)
        knee_delta[i] = math.degrees((R_knee.inv() * R_knee2).magnitude())
        for j, q in ((hip, hip_l), (knee, knee_l), (ankle, ankle_l3)):
            rot[i, j] = q.as_quat()[[3, 0, 1, 2]]
    return {'ik_frames': n_ik, 'clamped_frames': n_clamp, 'toe_err_max': float(toe_err.max()),
            'knee_delta_max_deg': float(knee_delta.max())}


# --------------------------------------------------------------------------------------- contact detection
def toe_speed(p):
    """Central-difference speed (units/frame), one-sided at the ends."""
    v = np.zeros(len(p))
    if len(p) > 2:
        v[1:-1] = np.linalg.norm(p[2:] - p[:-2], axis=-1) / 2.0
    if len(p) > 1:
        v[0] = np.linalg.norm(p[1] - p[0])
        v[-1] = np.linalg.norm(p[-1] - p[-2])
    return v


def majority_vote(c, window):
    if window <= 1:
        return c.copy()
    h = window // 2
    padded = np.pad(c.astype(int), h, mode='edge')
    return np.array([padded[i:i + window].sum() * 2 > window for i in range(len(c))])


def _runs(mask):
    """(start, end) of each run of True in a bool array (end exclusive)."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def clean_contacts(c, min_gap, min_len):
    """Fill gaps of <= min_gap frames between contacts, then drop contact runs shorter than min_len."""
    c = c.copy()
    for a, b in _runs(~c):
        if 0 < a and b < len(c) and b - a <= min_gap:
            c[a:b] = True
    for a, b in _runs(c):
        if b - a < min_len:
            c[a:b] = False
    return c


def detect_contacts(toe_pos, ground, vel_thr, height_thr, window, min_gap=2, min_len=3):
    v = toe_speed(toe_pos)
    h = toe_pos[:, 1] - ground
    c = (v < vel_thr) & (h < height_thr)
    return clean_contacts(majority_vote(c, window), min_gap, min_len), v, h


# -------------------------------------------------------------------------------- runtime foot locking (part 2)
def _cubic_weights(t, blend):
    w0 = 2 * t ** 3 - 3 * t ** 2 + 1
    w1 = (t ** 3 - 2 * t ** 2 + t) * blend
    w2 = (6 * t ** 2 - 6 * t) / max(blend, 1e-8)
    w3 = 3 * t ** 2 - 4 * t + 1
    return w0, w1, w2, w3


def inertialize_update(in_pos, in_vel, off_pos, off_vel, time, dt, blend):
    t = float(np.clip((time + dt) / max(blend, 1e-8), 0.0, 1.0))
    w0, w1, w2, w3 = _cubic_weights(t, blend)
    return in_pos + off_pos * w0 + off_vel * w1, in_vel + off_pos * w2 + off_vel * w3, time + dt


def inertialize_transition(off_pos, off_vel, time, src_pos, src_vel, dst_pos, dst_vel, blend):
    t = float(np.clip(time / max(blend, 1e-8), 0.0, 1.0))
    w0, w1, w2, w3 = _cubic_weights(t, blend)
    new_off_pos = (src_pos + off_pos * w0 + off_vel * w1) - dst_pos
    new_off_vel = (src_vel + off_pos * w2 + off_vel * w3) - dst_vel
    return new_off_pos, new_off_vel, 0.0


def foot_lock_runtime(toe_in, contact, dt, blend, lock_dist, unlock_dist, contact_height=None):
    """Per-frame toe target from the article's FootLockingState machine. Returns (targets, locked_flags)."""
    T = len(toe_in)
    zero = np.zeros(3)
    pos, vel = toe_in[0].copy(), zero.copy()
    in_pos, in_vel = toe_in[0].copy(), zero.copy()
    off_pos, off_vel, time = zero.copy(), zero.copy(), blend
    contact_pt, locked = zero.copy(), False
    out = np.zeros_like(toe_in)
    flags = np.zeros(T, dtype=bool)
    for i in range(T):
        in_vel = (toe_in[i] - in_pos) / max(dt, 1e-8) if i > 0 else zero.copy()
        in_pos = toe_in[i].copy()
        tgt_pos, tgt_vel = (contact_pt, zero) if locked else (in_pos, in_vel)
        pos, vel, time = inertialize_update(tgt_pos, tgt_vel, off_pos, off_vel, time, dt, blend)
        dist = np.linalg.norm(pos - in_pos)
        if not locked and contact[i] and dist < lock_dist:
            locked = True
            contact_pt = in_pos.copy()
            if contact_height is not None:
                contact_pt[1] = contact_height
            off_pos, off_vel, time = inertialize_transition(off_pos, off_vel, time, in_pos, in_vel, contact_pt, zero, blend)
        elif locked and (not contact[i] or dist > unlock_dist):
            locked = False
            off_pos, off_vel, time = inertialize_transition(off_pos, off_vel, time, contact_pt, zero, in_pos, in_vel, blend)
        out[i] = pos
        flags[i] = locked
    return out, flags


class FootLockStream:
    """`foot_lock(mode='runtime')` for a live stream: the same contact detection, article state machine and leg IK,
    one frame at a time with the state kept across calls (the realtime demo has no clip to loop over).

    Two differences forced by causality, both harmless at 30-60 fps: the contact vote can only look backwards (the
    clip version votes over a centred window and fills short gaps afterwards), and the toe speed is a backward
    difference rather than a central one. The vote window is given in 30 fps frames and rescaled to the caller's dt.

    Positions are in the config's units (cm by default); `scale` converts the caller's units into them, so the demo
    passes scale=100 for metres. Only rotations come back — runtime locking never moves the root — so this is a
    display-time filter: the stream's own frames stay exactly what the model produced, which is what the next
    chunk's history must be."""

    def __init__(self, offsets, parents, names, cfg=None, ground=0.0, scale=1.0, fps=30.0):
        self.Cfg = FootLockConfig.from_cfg(cfg)
        self.Parents = np.asarray(parents)
        self.Chains = resolve_chains(names)
        self.Scale = float(scale)
        self.Fps = float(fps)
        self.SetBody(offsets, ground)

    def SetBody(self, offsets, ground=0.0):
        """New skeleton (the beta sliders) or ground height; drops the lock state."""
        self.Offsets = np.asarray(offsets, np.float64) * self.Scale
        self.Ground = float(ground) * self.Scale
        needed = set()                                                            # the legs and their ancestors:
        for chain in self.Chains.values():                                        # 9 joints of 23, and this runs
            for joint in chain:                                                   # on every rendered frame
                j = int(joint)
                while j >= 0 and j not in needed:
                    needed.add(j)
                    j = int(self.Parents[j])
        self.Solved = sorted(needed)
        self.Reset()

    def _fk(self, rot, root):
        """`fk` restricted to `self.Solved` (parents come first, so one pass in index order is enough)."""
        pos = np.zeros((1, rot.shape[1], 3))
        glob = [None] * rot.shape[1]
        for j in self.Solved:
            p = int(self.Parents[j])
            local = R.from_quat(rot[:, j][:, [1, 2, 3, 0]])
            if p < 0:
                glob[j], pos[:, j] = local, root
            else:
                glob[j] = glob[p] * local
                pos[:, j] = pos[:, p] + glob[p].apply(self.Offsets[j][None])
        return pos, glob

    def Reset(self):
        zero = lambda: np.zeros(3)                                                # noqa: E731
        self.State = {k: {'pos': None, 'vel': zero(), 'in_pos': None, 'in_vel': zero(), 'off_pos': zero(),
                          'off_vel': zero(), 'time': self.Cfg.blend, 'contact_pt': zero(), 'locked': False,
                          'votes': []} for k in self.Chains}
        self.Stats = {'frames': 0, 'locked': 0, 'contact': 0, 'skate_in': 0.0, 'skate_out': 0.0, 'skate_frames': 0}

    def Skate(self):
        """Mean toe xz speed over the frames the feet were in contact, before and after the lock (cm/frame)."""
        n = max(self.Stats['skate_frames'], 1)
        return self.Stats['skate_in'] / n, self.Stats['skate_out'] / n

    def _step(self, key, toe, dt):
        st, cfg, zero = self.State[key], self.Cfg, np.zeros(3)
        if st['in_pos'] is None:
            st['in_pos'], st['pos'] = toe.copy(), toe.copy()
        else:
            st['in_vel'] = (toe - st['in_pos']) / max(dt, 1e-8)
            st['in_pos'] = toe.copy()
        speed = float(np.linalg.norm(st['in_vel'])) / self.Fps                    # units per 30 fps frame
        raw = speed < cfg.vel_thr and (toe[1] - self.Ground) < cfg.height_thr
        window = max(1, int(round(cfg.vote / max(dt * self.Fps, 1e-6))))          # cfg.vote is in 30 fps frames
        st['votes'].append(bool(raw))
        del st['votes'][:-window]
        # No lock until the window has filled: the first frame after a reset has no velocity yet, so a foot in
        # mid-swing would read as planted and snap to the ground the moment the lock is switched on.
        contact = len(st['votes']) >= window and 2 * sum(st['votes']) > window

        tgt_pos, tgt_vel = (st['contact_pt'], zero) if st['locked'] else (st['in_pos'], st['in_vel'])
        st['pos'], st['vel'], st['time'] = inertialize_update(tgt_pos, tgt_vel, st['off_pos'], st['off_vel'],
                                                              st['time'], dt, cfg.blend)
        dist = float(np.linalg.norm(st['pos'] - st['in_pos']))
        if not st['locked'] and contact and dist < cfg.lock_dist:
            st['locked'] = True
            st['contact_pt'] = st['in_pos'].copy()
            if cfg.plant:
                st['contact_pt'][1] = self.Ground
            st['off_pos'], st['off_vel'], st['time'] = inertialize_transition(
                st['off_pos'], st['off_vel'], st['time'], st['in_pos'], st['in_vel'], st['contact_pt'], zero, cfg.blend)
        elif st['locked'] and (not contact or dist > cfg.unlock_dist):
            st['locked'] = False
            st['off_pos'], st['off_vel'], st['time'] = inertialize_transition(
                st['off_pos'], st['off_vel'], st['time'], st['contact_pt'], zero, st['in_pos'], st['in_vel'], cfg.blend)
        self.Stats['contact'] += int(contact)
        self.Stats['locked'] += int(st['locked'])
        if contact and st.get('out_prev') is not None:
            per_frame = max(dt * self.Fps, 1e-8)                                  # render step -> 30 fps frames
            self.Stats['skate_in'] += float(np.linalg.norm((toe - st['in_prev'])[[0, 2]])) / per_frame
            self.Stats['skate_out'] += float(np.linalg.norm((st['pos'] - st['out_prev'])[[0, 2]])) / per_frame
            self.Stats['skate_frames'] += 1
        st['in_prev'], st['out_prev'] = toe.copy(), st['pos'].copy()
        return st['pos']

    def __call__(self, rot_wxyz, root, dt):
        """One frame: local quats (J,4) wxyz + root (3) in the caller's units -> new (J,4) with the legs solved."""
        rot = np.asarray(rot_wxyz, np.float64).reshape(1, -1, 4).copy()
        root = np.asarray(root, np.float64).reshape(1, 3) * self.Scale
        pos, orient = self._fk(rot, root)
        targets = {k: self._step(k, pos[0, chain[3]], dt) for k, chain in self.Chains.items()}
        for k, chain in self.Chains.items():
            solve_leg(rot, pos, orient, self.Offsets, self.Parents, chain, targets[k][None],
                      self.Cfg.max_ext, self.Cfg.softening)
        self.Stats['frames'] += 1
        return rot[0]


# ------------------------------------------------------------------------------- offline constraint solve (part 4)
def foot_lock_offline(toes_in, pelvis_in, contacts, soft=0.05, hard=0.9, iters=5000, plant_height=None, pin=0):
    """Position-based solve over the clip. toes_in: {'left': (T,3), 'right': (T,3)}; contacts: same keys, bool.
    Returns (toe targets dict, pelvis positions). Vectorised red/black ordering of the article's frame loop; the
    pelvis chain is pulled in both directions (the article's one-directional update lets the clip drift) and the
    first `pin` frames (the true history) are held at their source positions."""
    T = len(pelvis_in)
    toes = {k: v.copy() for k, v in toes_in.items()}
    pelvis = pelvis_in.copy()
    rest_len = {k: np.linalg.norm(pelvis_in - toes_in[k], axis=-1) for k in toes}
    for _ in range(iters):
        for parity in (1, 0):
            i = np.arange(1, T)[parity::2]
            for k in toes:
                X, Xin, c = toes[k], toes_in[k], contacts[k]
                both = (c[i - 1] & c[i])[:, None]
                mid = 0.5 * (X[i - 1] + X[i])
                if plant_height is not None:
                    mid[:, 1] = plant_height
                prev_t = X[i] + (Xin[i - 1] - Xin[i])
                curr_t = X[i - 1] + (Xin[i] - Xin[i - 1])
                new_prev = np.where(both, X[i - 1] + (mid - X[i - 1]) * hard, X[i - 1] + (prev_t - X[i - 1]) * soft)
                new_curr = np.where(both, X[i] + (mid - X[i]) * hard, X[i] + (curr_t - X[i]) * soft)
                X[i - 1], X[i] = new_prev, new_curr
            prev_t = pelvis[i] + (pelvis_in[i - 1] - pelvis_in[i])
            curr_t = pelvis[i - 1] + (pelvis_in[i] - pelvis_in[i - 1])
            new_prev = pelvis[i - 1] + (prev_t - pelvis[i - 1]) * soft
            new_curr = pelvis[i] + (curr_t - pelvis[i]) * soft
            pelvis[i - 1], pelvis[i] = new_prev, new_curr
        for k in toes:
            X = toes[k]
            direction = pelvis - X
            direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
            hip_prev = pelvis.copy()
            pelvis += ((X + direction * rest_len[k][:, None]) - pelvis) * soft
            X += ((hip_prev - direction * rest_len[k][:, None]) - X) * soft
        if pin > 0:
            pelvis[:pin] = pelvis_in[:pin]
            for k in toes:
                toes[k][:pin] = toes_in[k][:pin]
    return toes, pelvis


# ------------------------------------------------------------------------------------------------ metrics
def skate_metrics(toe_pos, contact):
    """Mean / max toe xz-speed (units/frame) over the given contact frames."""
    if contact.sum() == 0:
        return {'frames': 0, 'mean': None, 'max': None}
    v = toe_speed(toe_pos[:, [0, 2]])
    return {'frames': int(contact.sum()), 'mean': float(v[contact].mean()), 'max': float(v[contact].max())}


def max_accel(toe_pos):
    if len(toe_pos) < 3:
        return 0.0
    a = toe_pos[2:] - 2 * toe_pos[1:-1] + toe_pos[:-2]
    return float(np.linalg.norm(a, axis=-1).max())


def measure(rotations, root_pos, offsets, parents, names, cfg=None):
    """Contact detection + skating metrics of a clip, without modifying it. Returns (info, positions)."""
    cfg = FootLockConfig.from_cfg(cfg)
    rot = np.asarray(rotations, np.float64)
    pos, _ = fk(rot, np.asarray(root_pos, np.float64), np.asarray(offsets, np.float64), np.asarray(parents))
    chains = resolve_chains(names)
    toe_idx = {k: v[3] for k, v in chains.items()}
    ground = cfg.ground if cfg.ground is not None else min(pos[:, ti, 1].min() for ti in toe_idx.values())
    info = {'ground': float(ground), 'contacts': {}, 'speed': {}, 'height': {}, 'legs': {}}
    for k, ti in toe_idx.items():
        c, v, h = detect_contacts(pos[:, ti], ground, cfg.vel_thr, cfg.height_thr, cfg.vote, cfg.min_gap, cfg.min_len)
        info['contacts'][k], info['speed'][k], info['height'][k] = c, v, h
        info['legs'][k] = {'contact_frames': int(c.sum()), 'skate': skate_metrics(pos[:, ti], c), 'toe_max_accel': max_accel(pos[:, ti])}
    return info, pos


# ------------------------------------------------------------------------------------------------ driver
def foot_lock(rotations, root_pos, offsets, parents, names, frametime, cfg=None, pin=None, contacts=None):
    """Foot-lock one clip. rotations (T,J,4 wxyz), root_pos (T,3), offsets (J,3) in the skeleton's units.
    `pin` = number of leading history frames (used by the offline solve when cfg.pin is None).
    `contacts` = optional {'left': bool (T,), 'right': bool (T,)} to use instead of detecting them on this clip
    (e.g. the source clip's contacts when retargeting)."""
    cfg = FootLockConfig.from_cfg(cfg)
    rot = np.asarray(rotations, np.float64).copy()
    root = np.asarray(root_pos, np.float64).copy()
    offsets = np.asarray(offsets, np.float64)
    parents = np.asarray(parents)
    chains = resolve_chains(names)
    toe_idx = {k: v[3] for k, v in chains.items()}
    dt = float(frametime)

    pos, orient = fk(rot, root, offsets, parents)
    ground = cfg.ground if cfg.ground is not None else min(pos[:, ti, 1].min() for ti in toe_idx.values())
    given = contacts
    contacts, speeds, heights = {}, {}, {}
    for k, ti in toe_idx.items():
        contacts[k], speeds[k], heights[k] = detect_contacts(pos[:, ti], ground, cfg.vel_thr, cfg.height_thr, cfg.vote, cfg.min_gap, cfg.min_len)
        if given is not None:
            contacts[k] = np.asarray(given[k], dtype=bool)
    plant = ground if cfg.plant else None

    if cfg.mode == 'runtime':
        targets, locked = {}, {}
        for k, ti in toe_idx.items():
            targets[k], locked[k] = foot_lock_runtime(pos[:, ti], contacts[k], dt, cfg.blend, cfg.lock_dist, cfg.unlock_dist, contact_height=plant)
        new_root = root
    else:
        n_pin = cfg.pin if cfg.pin is not None else int(pin or 0)
        targets, new_root = foot_lock_offline({k: pos[:, ti] for k, ti in toe_idx.items()}, root, contacts,
                                              soft=cfg.soft, hard=cfg.hard, iters=cfg.iters, plant_height=plant, pin=n_pin)
        locked = {k: contacts[k].copy() for k in toe_idx}
        pos, orient = fk(rot, new_root, offsets, parents)             # root moved -> redo FK before the IK

    new_rot = rot.copy()
    legs = {}
    for k, chain in chains.items():
        legs[k] = solve_leg(new_rot, pos, orient, offsets, parents, chain, targets[k], cfg.max_ext, cfg.softening)

    pos_out, _ = fk(new_rot, new_root, offsets, parents)
    for k, ti in toe_idx.items():
        leg = legs[k]
        leg['contact_frames'] = int(contacts[k].sum())
        leg['locked_frames'] = int(locked[k].sum())
        leg['skate_in'] = skate_metrics(pos[:, ti], contacts[k])
        leg['skate_out'] = skate_metrics(pos_out[:, ti], contacts[k])
        leg['toe_max_accel_in'] = max_accel(pos[:, ti])
        leg['toe_max_accel_out'] = max_accel(pos_out[:, ti])
        tdev = np.linalg.norm(pos_out[:, ti] - pos[:, ti], axis=-1)
        leg['toe_dev_max_cm'] = float(tdev.max())
        leg['toe_dev_gt10cm_frames'] = int((tdev > 10).sum())
    dev = np.linalg.norm(pos_out - pos, axis=-1)
    info = {'mode': cfg.mode, 'ground': float(ground), 'legs': legs,
            'pose_dev_mean_cm': float(dev.mean()), 'pose_dev_max_cm': float(dev.max()),
            'contacts': contacts, 'locked': locked, 'speed_in': speeds, 'height_in': heights,
            'speed_out': {k: toe_speed(pos_out[:, ti]) for k, ti in toe_idx.items()},
            'height_out': {k: pos_out[:, ti, 1] - ground for k, ti in toe_idx.items()},
            'positions_in': pos, 'positions_out': pos_out, 'toe_idx': toe_idx}
    return FootLockResult(rotations=new_rot, root_pos=new_root, info=info)


def foot_lock_motion(motion, cfg=None, pin=None):
    """Foot-lock a ``Motion`` (in its original units, i.e. offsets / scaling_factor). Returns (new Motion, info)."""
    scale = float(motion.scaling_factor) if motion.scaling_factor else 1.0
    res = foot_lock(motion.rotations, motion.positions[:, 0] / scale, motion.offsets / scale, motion.parents, motion.names,
                    motion.frametime, cfg, pin=pin)
    out = motion.copy()
    out.rotations = res.rotations.astype(motion.rotations.dtype)
    positions = out.positions.copy()
    positions[:, 0] = (res.root_pos * scale).astype(positions.dtype)
    out.positions = positions
    return out, res.info


def summarize(infos, gt_infos=None):
    """JSON-able aggregate of ``foot_lock`` info dicts (and optionally ``measure`` infos of the matching gt clips)."""
    def mean(vals):
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None
    legs = lambda info, key: [info['legs'][s][key] for s in ('left', 'right')]
    out = {
        'n_clips': len(infos),
        'skate_in': mean([l['mean'] for i in infos for l in legs(i, 'skate_in')]),
        'skate_out': mean([l['mean'] for i in infos for l in legs(i, 'skate_out')]),
        'contact_frames': int(sum(sum(legs(i, 'contact_frames')) for i in infos)),
        'locked_frames': int(sum(sum(legs(i, 'locked_frames')) for i in infos)),
        'clamped_frames': int(sum(sum(legs(i, 'clamped_frames')) for i in infos)),
        'toe_dev_max_cm': max([max(legs(i, 'toe_dev_max_cm')) for i in infos], default=None),
        'toe_dev_gt10cm_frames': int(sum(sum(legs(i, 'toe_dev_gt10cm_frames')) for i in infos)),
        'knee_delta_max_deg': max([max(legs(i, 'knee_delta_max_deg')) for i in infos], default=None),
        'pose_dev_mean_cm': mean([i['pose_dev_mean_cm'] for i in infos]),
        'pose_dev_max_cm': max([i['pose_dev_max_cm'] for i in infos], default=None),
    }
    if gt_infos:
        out['skate_gt'] = mean([i['legs'][s]['skate']['mean'] for i in gt_infos for s in ('left', 'right')])
    return out
