"""The predefined one-minute evaluation route (paper Fig. 6).

One file, `data/eval/eval_traj.npz`, is the only source of the command sequence. Everything is in LEG-LENGTH units:
positions in legs, speeds in legs per second; a body multiplies by its `leg` (data/bodies/bodies.csv), so one command
is "fixed in body-relative units on every body" (paper 6.1).

Rows: row 0 is the origin = the pivot of block 0 (the canonical history's last frame), rows 1..n_eval are the evaluated
frames, rows n_eval+1..n_eval+n_lookahead are the look-ahead the last block needs (block b's plan is rows
hop*b+1 .. hop*b+F, the 45 frames AFTER its pivot frame -- the training convention, data/loco_dataset.py get_raw_item
slices traj_*[K:] relative to frame K-1, and realtime/model_loop.py::PersonaModel.generate).

Angles: yaw about +Y with +Z = 0, `yaw = atan2(dx, dz)` (realtime/trajectory.py::yaw_of, make_pkl.extract_traj);
`traj_quat` = yaw_quat(facing) = (cos(y/2), 0, sin(y/2), 0) wxyz. `heading` is the direction of travel, `facing` the
commanded body facing; they differ only in the side-step and backward segments.

Per-block conditioning (block_plan): traj_xz = route_xz[rows] - actual root xz of the last committed frame (world
axes, aug theta = 0), traj_quat = yaw_quat(facing[rows]). The route is world-fixed and time-indexed: the model is
pulled back toward the route whenever it drifts, and the tracking metric measures exactly that drift.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 30
N_EVAL = 1800          # 60 s
N_LOOKAHEAD = 45       # one block of plan beyond the last pivot
VERSION = 'eval_traj_v1'
SEGMENTS = ('walk', 'fastwalk', 'turn90', 'run', 'arc90', 'sharp135', 'stop', 'side', 'back', 'meander', 'uturn', 'run2', 'end')


def yaw_quat(yaw):
    """Yaw angles (rad, +Z = 0) -> wxyz quaternions rotating +Z onto the direction (training traj_quat convention)."""
    yaw = np.asarray(yaw, np.float64)
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.sin(yaw / 2), np.zeros_like(yaw)], -1)


def yaw_of(direction):
    d = np.asarray(direction, np.float64)
    return np.arctan2(d[..., 0], d[..., 1])


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def keyed_profile(t, keys):
    """Piecewise profile through (time, value) keys with a smoothstep (zero-slope) blend between consecutive keys
    that differ in value; constant between equal keys. C1 everywhere."""
    t = np.asarray(t, np.float64)
    out = np.full_like(t, float(keys[0][1]))
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        m = (t >= t0) & (t < t1)
        if t1 > t0:
            out[m] = v0 + (v1 - v0) * smoothstep((t[m] - t0) / (t1 - t0))
    out[t >= keys[-1][0]] = float(keys[-1][1])
    return out


def raised_cosine_bump(t, t0, dur, total):
    """Yaw-rate bump whose integral over [t0, t0+dur] is `total` (rad): (total/dur) * (1 - cos(2 pi u)) / 1, u in [0,1]."""
    u = (np.asarray(t, np.float64) - t0) / dur
    rate = np.where((u >= 0) & (u < 1), (total / dur) * (1.0 - np.cos(2.0 * np.pi * u)), 0.0)
    return rate


@dataclass(frozen=True)
class Route:
    xz: np.ndarray        # (N, 2) legs, row 0 = origin
    speed: np.ndarray     # (N,) legs / s, speed[r] = |xz[r+1] - xz[r]| * fps
    heading: np.ndarray   # (N,) rad
    facing: np.ndarray    # (N,) rad
    segment: np.ndarray   # (N,) int, index into SEGMENTS
    fps: int
    n_eval: int
    n_lookahead: int
    v0_leg: float
    version: str
    sha256: str = ''
    segments_json: str = ''

    @property
    def n(self):
        return int(self.xz.shape[0])

    @property
    def quat(self):
        return yaw_quat(self.facing)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, xz=self.xz, speed=self.speed, heading=self.heading, facing=self.facing, segment=self.segment,
                 fps=self.fps, n_eval=self.n_eval, n_lookahead=self.n_lookahead, v0_leg=self.v0_leg,
                 version=self.version, segments=self.segments_json, segment_names=np.array(SEGMENTS))
        return file_sha256(path)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(xz=z['xz'], speed=z['speed'], heading=z['heading'], facing=z['facing'], segment=z['segment'],
                   fps=int(z['fps']), n_eval=int(z['n_eval']), n_lookahead=int(z['n_lookahead']), v0_leg=float(z['v0_leg']),
                   version=str(z['version']), sha256=file_sha256(path), segments_json=str(z['segments']))

    def scaled(self, leg, v_cap_leg=None):
        """Metres for one body: positions and speeds x leg; if the route's top speed exceeds the body's cap (legs/s),
        the WHOLE route is scaled down uniformly (recorded as `scale`), facing untouched."""
        scale = 1.0
        vmax = float(self.speed.max())
        if v_cap_leg is not None and vmax > v_cap_leg:
            scale = float(v_cap_leg / vmax)
        return ScaledRoute(xz_m=self.xz * leg * scale, speed_mps=self.speed * leg * scale, heading=self.heading,
                           facing=self.facing, quat=self.quat, leg=float(leg), scale=scale,
                           v_cap_leg=None if v_cap_leg is None else float(v_cap_leg))


@dataclass(frozen=True)
class ScaledRoute:
    xz_m: np.ndarray
    speed_mps: np.ndarray
    heading: np.ndarray
    facing: np.ndarray
    quat: np.ndarray
    leg: float
    scale: float
    v_cap_leg: float | None


def block_rows(block, hop=3, F=45):
    """Plan rows of block b: the F frames after its pivot frame (frame hop*b, row hop*b)."""
    return slice(hop * block + 1, hop * block + 1 + F)


def block_plan(xz_m, quat, pivot_xz, block, hop=3, F=45):
    """xz_m (B, N, 2) metres, quat (B, N, 4), pivot_xz (B, 2) = the actual root xz of the last committed frame.
    -> traj_xz (B, F, 2) relative to the pivot in world axes, traj_quat (B, F, 4)."""
    rows = block_rows(block, hop, F)
    return xz_m[:, rows] - pivot_xz[:, None, :], quat[:, rows]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ the design
def design(v0):
    """The segment table of the route, in absolute legs/s for a canonical walking speed v0.
    Returns (speed keys, yaw-rate bumps, facing offsets, meander, segment spans)."""
    # top speed under every body's p95 cap (data/eval/speed_caps.csv: 1.97-2.00 legs/s on all 128 bodies), so no body
    # is ever scaled: run = 1.45 v0 = 1.90 legs/s (p02 1.48 m/s; the data's running takes average 1.3-1.5 legs/s)
    fast, run, side, back, slow = 1.25 * v0, 1.45 * v0, 0.9 * v0, 0.87 * v0, 0.55 * v0
    speed_keys = [(0.0, v0), (4.0, v0), (6.5, fast), (12.0, fast), (14.5, run), (20.0, run), (22.5, v0), (24.0, v0),
                  (26.0, 0.0), (27.5, 0.0), (29.0, v0), (29.5, v0), (30.5, side), (34.5, side), (35.5, back), (39.5, back),
                  (41.0, slow), (48.0, slow), (49.0, v0), (52.0, v0), (53.5, run), (56.0, run), (57.5, v0), (61.5, v0)]
    # (t0, duration, total angle in rad); positive = left turn (yaw increases toward +X from +Z)
    bumps = [(8.0, 2.0, np.radians(90)), (16.0, 4.0, np.radians(-90)), (21.0, 1.5, np.radians(135)),
             (29.5, 1.0, np.radians(90)), (33.0, 1.0, np.radians(-90)), (48.0, 2.5, np.radians(180))]
    facing = {'side': (29.5, 34.0), 'back': (35.5, 39.5, 0.75)}     # side: facing frozen; back: +180 deg with 0.75 s ramps
    meander = (40.0, 48.0, np.radians(40), 4.0, 1.0)                 # (t0, t1, amplitude, period, edge window)
    spans = [(0, 4, 'walk'), (4, 8, 'fastwalk'), (8, 12, 'turn90'), (12, 16, 'run'), (16, 20, 'arc90'), (20, 24, 'sharp135'),
             (24, 29, 'stop'), (29, 35, 'side'), (35, 40, 'back'), (40, 48, 'meander'), (48, 52, 'uturn'), (52, 56, 'run2'),
             (56, 62, 'end')]
    return speed_keys, bumps, facing, meander, spans


def build_route(v0_leg, fps=FPS, n_eval=N_EVAL, n_lookahead=N_LOOKAHEAD, version=VERSION):
    n = 1 + n_eval + n_lookahead
    t = np.arange(n, dtype=np.float64) / fps
    speed_keys, bumps, facing_spec, meander, spans = design(float(v0_leg))
    speed = keyed_profile(t, speed_keys)
    rate = np.zeros(n)
    for t0, dur, total in bumps:
        rate += raised_cosine_bump(t, t0, dur, total)
    heading = np.concatenate([[0.0], np.cumsum(rate[:-1]) / fps])          # heading[0] = 0 = +Z
    m0, m1, amp, period, edge = meander
    u = (t - m0)
    win = smoothstep(u / edge) * smoothstep((m1 - t) / edge)
    heading = heading + np.where((t >= m0) & (t <= m1), amp * np.sin(2 * np.pi * u / period) * win, 0.0)
    # facing: side segment freezes it, back segment adds +180 deg with ramps
    facing = heading.copy()
    s0, s1 = facing_spec['side']
    m = (t >= s0) & (t <= s1)
    h_ref = np.interp(s0, t, heading)
    facing[m] = h_ref
    b0, b1, ramp = facing_spec['back']
    off = np.pi * (smoothstep((t - b0) / ramp) - smoothstep((t - (b1 - ramp)) / ramp))
    facing = facing + off
    # integrate the position
    step = speed[:-1] / fps
    d = np.stack([np.sin(heading[:-1]), np.cos(heading[:-1])], -1) * step[:, None]
    xz = np.concatenate([np.zeros((1, 2)), np.cumsum(d, axis=0)])
    seg = np.zeros(n, dtype=np.int64)
    for a, b, name in spans:
        seg[(t >= a) & (t < b)] = SEGMENTS.index(name)
    seg[t >= spans[-1][1]] = SEGMENTS.index('end')
    r = Route(xz=xz, speed=speed, heading=heading, facing=facing, segment=seg, fps=fps, n_eval=n_eval,
              n_lookahead=n_lookahead, v0_leg=float(v0_leg), version=version,
              segments_json=json.dumps({'speed_keys': speed_keys, 'bumps_deg': [(a, b, float(np.degrees(c))) for a, b, c in bumps],
                                        'facing': facing_spec, 'meander_deg': [m0, m1, float(np.degrees(amp)), period, edge], 'spans': spans}))
    return r

def build_straight_route(v_leg, fps=FPS, n_eval=N_EVAL, n_lookahead=N_LOOKAHEAD, version='straight-v1', speed_keys=None,
                         spans=None):
    """A straight route at one constant speed (legs/s), heading = facing = +Z: the paper's tab:guidance scenario
    (neutral -> bigstep at 1.6 m/s, the style switched mid-route by the engine). Same file layout as build_route.

    Optional (default off, the constant route is unchanged bit for bit): `speed_keys` = [(t s, legs/s), ...] gives a
    keyed speed profile (keyed_profile, smoothstep between keys) along the same straight line, and `spans` =
    [(t0, t1, segment name), ...] labels it -- used by build_video_straight_route."""
    n = 1 + n_eval + n_lookahead
    heading = np.zeros(n); facing = np.zeros(n)
    if speed_keys is None:
        speed = np.full(n, float(v_leg))
        xz = np.stack([np.zeros(n), np.arange(n, dtype=np.float64) * float(v_leg) / fps], -1)
        seg = np.full(n, SEGMENTS.index('walk'), dtype=np.int64)
        segments_json = '{"straight": [0, %d]}' % n
    else:
        t = np.arange(n, dtype=np.float64) / fps
        speed = keyed_profile(t, speed_keys)
        xz = np.stack([np.zeros(n), np.concatenate([[0.0], np.cumsum(speed[:-1]) / fps])], -1)   # speed[r] = |xz[r+1]-xz[r]| fps
        seg = np.full(n, SEGMENTS.index('walk'), dtype=np.int64)
        for a, b, name in spans or ():
            seg[(t >= a) & (t < b)] = SEGMENTS.index(name)
        if spans:
            seg[t >= spans[-1][1]] = SEGMENTS.index(spans[-1][2])
        segments_json = json.dumps({'straight': [0, n], 'speed_keys': [list(map(float, k)) for k in speed_keys], 'spans': spans or []})
    return Route(xz=xz, speed=speed, heading=heading, facing=facing, segment=seg, fps=fps, n_eval=n_eval,
                 n_lookahead=n_lookahead, v0_leg=float(v_leg), version=version, segments_json=segments_json)


# the paper video's single-axis route: straight along +Z, 20 s of steady walking at the
# canonical v0, a 1 s smoothstep ramp, 5 s of running at the evaluation route's run speed (1.45 v0 = 1.90 legs/s, under
# every body's p95 cap 1.97-2.00 legs/s, so no body is ever scaled). 26 s = 780 frames = 65 blocks at hop 12, 260 at hop 3.
VIDEO_STRAIGHT = dict(walk_s=20.0, ramp_s=1.0, run_s=5.0, run_mult=1.45, version='video_straight-v1')


def build_video_straight_route(v0_leg, fps=FPS, n_lookahead=N_LOOKAHEAD, walk_s=VIDEO_STRAIGHT['walk_s'], ramp_s=VIDEO_STRAIGHT['ramp_s'],
                               run_s=VIDEO_STRAIGHT['run_s'], run_mult=VIDEO_STRAIGHT['run_mult'], version=VIDEO_STRAIGHT['version']):
    """`video_straight` (scripts/make_eval_traj.py video): build_straight_route with a walk -> run speed profile.
    Starts at v0 like the evaluation route, so the canonical history (speed-matched to v0) joins it the same way."""
    v0 = float(v0_leg); run = run_mult * v0
    t_run = walk_s + ramp_s
    n_eval = int(round((t_run + run_s) * fps))
    keys = [(0.0, v0), (walk_s, v0), (t_run, run), (t_run + run_s + n_lookahead / fps + 1.0, run)]
    spans = [(0.0, walk_s, 'walk'), (walk_s, t_run + run_s, 'run')]
    return build_straight_route(v0, fps=fps, n_eval=n_eval, n_lookahead=n_lookahead, version=version, speed_keys=keys, spans=spans)
