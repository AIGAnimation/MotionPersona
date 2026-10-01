"""Root trajectory controller laid out like the model's trajectory conditioning.

Series layout: TimeSeries(0, 2.5, 76) -> sample 0 = now, sample i = i frames ahead at 30 fps; samples p+1..p+45 are
the `traj_xz` / `traj_quat` the model was trained on (data/loco_dataset.py: futures relative to frame K-1) for a
chunk continuing from the frame p frames ahead (p = the stream's buffered lead, 0 <= p <= LEAD_MAX).

Differences to the stock ai4animationpy Biped demo controller (numbers measured against the training trajectories):
  - the pivot is anchored to the actual actor root every frame (Synchronization = 1), so the plan always starts where
    the character is and the model is never asked to jump onto a diverged simulation line;
  - move / turn sensitivity 8 (time constant 0.125 s) instead of 10: 10 gives 10x the accelerations and 5-10x the
    yaw rates of the GT trajectories; 3 matches the GT statistics best but feels sticky in the closed loop (character
    t50 for idle->run 0.83 s with a deeper chunk buffer), 5 + the 3-frame hop gives t50 0.28 s / turn t90 0.9 s and
    8 (the default) 0.25 s / 0.7 s;
  - the model's own predicted future root path (SetPrediction) can be blended in Biped-style (TrajectoryCorrection):
    off by default — with the stream regenerating every few frames from its own history the seams are already as
    smooth as a normal step (seam/step ratio 0.95), and the blend only adds 0.05-0.1 s of input lag;
  - Conditioning(start) emits traj_xz / traj_quat with exactly the training convention (world axes by default).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))

from ai4animation import AI4Animation, RootModule, Rotation, Tensor, TimeSeries, Transform, Vector3

FPS = 30
FUTURE = 45
# The chunk model is asked for the next chunk while the stream still has up to LEAD_MAX frames buffered ahead of the
# playhead (model_loop.ChunkStreamer), so the plan must extend LEAD_MAX frames past the 45-frame window the model sees.
LEAD_MAX = 30
TRAJECTORY_WINDOW = (FUTURE + LEAD_MAX) / FPS
TRAJECTORY_SAMPLES = FUTURE + LEAD_MAX + 1       # sample 0 = now
DRAW_STRIDE = 5                                  # marker every 5 frames (0.17 s): 16 markers over the 2.5 s plan
DRAW_ARROW = 0.25                                # facing arrow length (m)
MOVE_SENSITIVITY = 8.0            # chosen in closed-loop feel tests; 5 costs 0.2 s on a stop
TURN_SENSITIVITY = 8.0            # only used when TURN_RATE = 0 (the stock exponential turn)
TURN_RATE = 120.0                 # deg/s, the plan's heading slew limit -- see ramp_planar; measured closed-loop
                                  # turn t90 0.74 s at 120 vs 0.79 at 100
TRAJECTORY_CORRECTION = 0.0
SYNCHRONIZATION = 1.0
LERP_EPS = 0.01                                  # Vector3.LerpDt / SlerpDt snap to the target within this distance


def yaw_of(direction):
    """Planar direction -> yaw angle (rad) about +Y with +Z = 0 (make_pkl.extract_traj convention)."""
    return float(np.arctan2(direction[0], direction[2]))


def yaw_quat(yaw):
    """Yaw angles -> wxyz quaternions rotating +Z onto the direction (training traj_quat convention)."""
    yaw = np.asarray(yaw, np.float64)
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.sin(yaw / 2), np.zeros_like(yaw)], -1)


def slerp_planar(start, end, weights):
    """`Vector3.Slerp(start, end, w)` for a whole array of weights at once (the library call is per-sample: 75 of them
    per rendered frame is 4.6 ms of the main thread, which is GIL the inference thread does not get).  Same formula,
    minus the exactly-180-degrees random jitter — `_resolve_direction` nudges that case away before we get here."""
    start = np.ravel(np.asarray(start, np.float64))[:3]
    end = np.ravel(np.asarray(end, np.float64))[:3]
    start = start / (np.linalg.norm(start) or 1.0)                     # Tensor.Normalize: a zero vector stays zero
    end = end / (np.linalg.norm(end) or 1.0)
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    rel = end - start * dot
    norm = float(np.linalg.norm(rel))
    rel = rel / norm if norm > 0.0 else rel
    theta = np.arccos(dot) * np.asarray(weights, np.float64)
    return start[None, :] * np.cos(theta)[:, None] + rel[None, :] * np.sin(theta)[:, None]


def ramp_planar(start, end, times, rate):
    """Heading ramp at a constant `rate` (deg/s): the shape a real turn actually has.

    Measured on the training data (180k windows): a real turn is a straight yaw ramp --
    the median normalised curve is within 1.7 % of a straight line -- at a per-frame rate whose p99 is 172 deg/s
    and whose largest value anywhere is 547 deg/s. `Vector3.Slerp(dir0, target, i/FUTURE)` after a `SlerpDt` step
    whips the PIVOT instead: at sens 8 the plan's base heading rotates 632 deg/s between frames (above the data's
    single-frame max of 547) while the window's own per-frame rate stays a gentle 46 deg/s, so every chunk's
    conditioning arrives ~21 deg rotated from the previous one.
    `times` = seconds from now for each sample; the ramp stops on the target and stays there."""
    start = np.ravel(np.asarray(start, np.float64))[:3]
    end = np.ravel(np.asarray(end, np.float64))[:3]
    if np.linalg.norm(start) == 0.0 or np.linalg.norm(end) == 0.0:
        return np.tile(start if np.linalg.norm(start) else end, (len(np.atleast_1d(times)), 1))
    yaw0, yaw1 = yaw_of(start), yaw_of(end)
    delta = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi                # shortest way round
    step = np.minimum(np.abs(delta), np.radians(rate) * np.asarray(times, np.float64))
    yaw = yaw0 + np.sign(delta) * step
    return np.stack([np.sin(yaw), np.zeros_like(yaw), np.cos(yaw)], -1)


def to_frame(xz, yaw):
    """Express planar (N,2) world xz vectors in the frame whose +Z points at `yaw` (rotation by -yaw about +Y)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * xz[:, 0] - s * xz[:, 1], s * xz[:, 0] + c * xz[:, 1]], -1)


class TrajectoryController:
    def __init__(self, window=TRAJECTORY_WINDOW, samples=TRAJECTORY_SAMPLES, move=MOVE_SENSITIVITY, turn=TURN_SENSITIVITY,
                 correction=TRAJECTORY_CORRECTION, synchronization=SYNCHRONIZATION, turn_rate=TURN_RATE, steer=False):
        self.Sim = RootModule.Series(TimeSeries(0.0, window, samples))      # raw spring-damper controller
        self.Plan = RootModule.Series(TimeSeries(0.0, window, samples))     # what the model is conditioned on
        self.Prediction = None                                              # RootModule.Series from the model loop
        self.MoveSensitivity = move
        self.TurnSensitivity = turn
        self.TurnRate = float(turn_rate or 0.0)      # deg/s; 0 = the stock exponential turn (TurnSensitivity)
        self.Correction = correction
        self.Synchronization = synchronization
        # steer (off by default): with the facing unlocked, the plan's velocity turns WITH its facing (rate-limited
        # ramp, speed eased on its own) instead of lerping straight to the stick vector, so heading == facing through
        # a turn and the commanded path is an arc -- the eval routes' shape. Off, the velocity swings to a new stick
        # direction in ~0.2 s while the facing slews at TurnRate: the plan crabs through every corner.
        self.Steer = bool(steer)

    # ----------------------------------------------------------------------------- per-frame control
    def Control(self, root_position, direction, velocity, dt, root_forward=None, facing_leash=0.0):
        """Advance the controller one frame. root_position = the actor's actual root (ground point), direction = facing
        setpoint (zero vector = face the velocity), velocity = planar velocity setpoint [m/s].

        `facing_leash` > 0 (deg, off by default) keeps the plan's facing within that angle of the character's own
        facing `root_forward`: the position is anchored to the actor every frame but the facing is the controller's
        own state, so after a turn input the plan faces the new way long before the body does (40 deg apart at walking
        speed), and the model is asked for a facing jump no training window contains (the first future sample is
        within 6 deg of the current heading at p99) -- it then turns slowly and walks sideways meanwhile."""
        position = Vector3.Lerp(self.Sim.GetPosition(0), root_position, self.Synchronization)
        if root_forward is not None and facing_leash > 0.0:
            self._leash(root_forward, facing_leash)
        steering = self.Steer and self.TurnRate > 0.0 and float(np.linalg.norm(np.ravel(direction))) == 0.0
        direction = self._resolve_direction(direction, velocity)
        if steering:
            self._steer(position, direction, velocity, dt)
        else:
            self._control(position, direction, velocity, dt)

        if self.Prediction is None or self.Correction <= 0.0:
            self.Plan.Transforms = self.Sim.Transforms.copy()
            self.Plan.Velocities = self.Sim.Velocities.copy()
            return
        # Biped-style correction: pull the plan towards the model's own continuation so the seam stays smooth
        self.Plan.Transforms = Transform.Interpolate(self.Sim.Transforms, self.Prediction.Transforms, self.Correction)
        current = np.asarray(root_position, np.float64).reshape(-1, 3)
        positions = Transform.GetPosition(self.Plan.Transforms)
        for i in range(1, self.Plan.SampleCount):
            target = positions[i:]
            time = self.Plan.Timestamps[i:].reshape(-1, 1)
            self.Plan.Velocities[i] = Tensor.Sum(target - current, axis=0, keepDim=False) / Tensor.Sum(time, axis=0, keepDim=False)
        self.Plan.Velocities[0] = self.Sim.Velocities[0]
        self.Plan.Velocities = Vector3.Lerp(self.Plan.Velocities, self.Prediction.Velocities, self.Correction)

    def _leash(self, root_forward, limit_deg):
        """Pull the plan's current facing to within `limit_deg` of the body's facing (Control's facing_leash)."""
        cur = np.ravel(np.asarray(self.Sim.GetDirection(0), np.float64))[:3]
        body = np.ravel(np.asarray(root_forward, np.float64))[:3]
        if np.linalg.norm(cur) == 0.0 or np.linalg.norm(body[[0, 2]]) == 0.0:
            return
        yc, yb = yaw_of(cur), yaw_of(body)
        d = (yc - yb + np.pi) % (2 * np.pi) - np.pi
        lim = np.radians(limit_deg)
        if abs(d) > lim:
            y = yb + np.sign(d) * lim
            self.Sim.SetDirection(np.array([np.sin(y), 0.0, np.cos(y)]), 0)

    def _control(self, position, direction, velocity, dt):
        """Series.Control with the sample ramp tied to the model's 45-frame window instead of the series length: the
        stock ramp `ratio = i / (SampleCount-1)` would, with the LEAD_MAX extension, let the model see only 60 % of a
        turn (`Slerp(dir0, target, ratio)` at sample 45) and a 1.7x slower velocity ramp; beyond FUTURE the samples
        continue at the setpoint.

        Samples 1.. are integrated in closed form over the whole series instead of sample by sample.  The scalar
        version (kept as the oracle in tests/test_model_loop.py) costs 4.6 ms of every rendered frame — Python and
        tiny-array numpy, i.e. 4.6 ms per frame during which the chunk thread cannot have the GIL, which is why a
        17 ms chunk took ~50 ms inside the running demo."""
        sim = self.Sim
        sim.SetVelocity(Vector3.LerpDt(sim.GetVelocity(0), velocity, dt, self.MoveSensitivity), 0)
        sim.SetPosition(position + sim.GetVelocity(0) * dt, 0)
        if self.TurnRate > 0.0:
            sim.SetDirection(ramp_planar(sim.GetDirection(0), direction, [dt], self.TurnRate)[0], 0)
        else:
            sim.SetDirection(Vector3.SlerpDt(sim.GetDirection(0), direction, dt, self.TurnSensitivity), 0)

        step = sim.DeltaTime
        ratio = np.minimum(np.arange(1, sim.SampleCount) / FUTURE, 1.0)
        target = np.ravel(np.asarray(velocity, np.float64))[:3]
        v0 = np.ravel(np.asarray(sim.GetVelocity(0), np.float64))[:3]
        # v[i] = LerpDt(v[i-1], target, step, ratio_i * sens) = target + (v0 - target) * exp(-step * sens * sum ratio),
        # and LerpDt snaps to the target once the gap is below LERP_EPS: the gap decays monotonically, so zeroing the
        # tail of the closed form is the same thing.
        gap = (v0 - target)[None, :]
        if self.MoveSensitivity != 0.0:
            gap = gap * np.exp(-step * self.MoveSensitivity * np.cumsum(ratio))[:, None]
            gap[np.linalg.norm(gap, axis=1) < LERP_EPS] = 0.0
        else:
            gap = np.repeat(gap, len(ratio), axis=0)                   # rate 0: LerpDt returns the previous value
        velocities = target[None, :] + gap
        positions = np.ravel(np.asarray(sim.GetPosition(0), np.float64))[:3] + step * np.cumsum(velocities, axis=0)
        if self.TurnRate > 0.0:
            directions = ramp_planar(sim.GetDirection(0), direction, np.arange(1, sim.SampleCount) * step, self.TurnRate)
        else:
            directions = slerp_planar(sim.GetDirection(0), direction, ratio)   # stock Biped: Slerp(dir0, .., i/FUTURE)

        sim.Velocities[1:] = velocities
        Transform.SetPosition(sim.Transforms[1:], positions)
        Transform.SetRotation(sim.Transforms[1:], Rotation.LookPlanar(directions))

    def _steer(self, position, direction, velocity, dt):
        """_control for `steer`: the facing ramps to the stick direction at TurnRate as in _control, and the velocity
        is the eased speed along that facing, now and for every future sample (speed: the closed-form LerpDt of the
        magnitude, same sensitivity; direction: the facing ramp)."""
        sim = self.Sim
        step = sim.DeltaTime
        target = float(np.linalg.norm(np.ravel(np.asarray(velocity, np.float64))[:3]))
        v_prev = float(np.linalg.norm(np.ravel(np.asarray(sim.GetVelocity(0), np.float64))[:3]))
        rate = self.MoveSensitivity
        speed0 = v_prev if rate == 0.0 else target + (v_prev - target) * np.exp(-dt * rate)
        if abs(speed0 - target) < LERP_EPS:
            speed0 = target
        dir0 = ramp_planar(sim.GetDirection(0), direction, [dt], self.TurnRate)[0]
        sim.SetDirection(dir0, 0)
        sim.SetVelocity(speed0 * dir0, 0)
        sim.SetPosition(position + sim.GetVelocity(0) * dt, 0)
        ratio = np.minimum(np.arange(1, sim.SampleCount) / FUTURE, 1.0)
        gap = np.full(len(ratio), speed0 - target)
        if rate != 0.0:
            gap = gap * np.exp(-step * rate * np.cumsum(ratio))
            gap[np.abs(gap) < LERP_EPS] = 0.0
        directions = ramp_planar(dir0, direction, np.arange(1, sim.SampleCount) * step, self.TurnRate)
        velocities = (target + gap)[:, None] * directions
        positions = np.ravel(np.asarray(sim.GetPosition(0), np.float64))[:3] + step * np.cumsum(velocities, axis=0)
        sim.Velocities[1:] = velocities
        Transform.SetPosition(sim.Transforms[1:], positions)
        Transform.SetRotation(sim.Transforms[1:], Rotation.LookPlanar(directions))

    def _resolve_direction(self, direction, velocity):
        """Series.Control's facing fallback (zero direction = face the velocity, idle = keep the current facing), plus a
        tie-break for a target exactly opposite to the current facing, where the planar slerp is degenerate and the
        character would never turn."""
        d = np.ravel(np.asarray(direction, np.float64))
        cur = np.ravel(np.asarray(self.Sim.GetDirection(0), np.float64))
        if np.linalg.norm(d) == 0.0:
            d = np.ravel(np.asarray(velocity, np.float64))
            if np.linalg.norm(d) == 0.0:
                # Idle with the facing unlocked: hold the facing. A zero target would decay the sim facing to zero within
                # a few frames (SlerpDt snaps to the target), which conditions the model on "face +Z" and gives the
                # model-less body a singular root transform.
                return Vector3.Create(*cur) if np.linalg.norm(cur) > 0.0 else Vector3.Create(0.0, 0.0, 1.0)
        d = d / np.linalg.norm(d)
        if np.dot(cur, d) < -0.9995 * np.linalg.norm(cur):
            a = np.deg2rad(2.0)                                            # nudge 2 degrees; the turn resolves it
            d = np.array([d[0] * np.cos(a) + d[2] * np.sin(a), 0.0, -d[0] * np.sin(a) + d[2] * np.cos(a)])
        return Vector3.Create(d[0], 0.0, d[2])

    def SetPrediction(self, positions, directions, velocities=None):
        """Feed the model's own future root path (world space, same layout: index 0 = now, i = i frames ahead). Fewer
        rows than samples are fine — the stream only knows its buffered frames — the rest copies the simulation, so
        the correction blend is a no-op there."""
        positions = np.asarray(positions, np.float64).reshape(-1, 3)
        directions = np.asarray(directions, np.float64).reshape(-1, 3)
        n = min(len(positions), len(directions), self.Sim.SampleCount)
        if n < 2:
            self.Prediction = None
            return
        transforms = self.Sim.Transforms.copy()
        transforms[:n] = Transform.TR(positions[:n], Rotation.LookPlanar(directions[:n]))
        vel = self.Sim.Velocities.copy()
        vel[:n] = np.gradient(positions[:n], self.Sim.DeltaTime, axis=0) if velocities is None else np.asarray(velocities, np.float64)[:n]
        self.Prediction = RootModule.Series(TimeSeries(self.Sim.Start, self.Sim.End, self.Sim.SampleCount), transforms, vel)

    def ClearPrediction(self):
        self.Prediction = None

    # ----------------------------------------------------------------------------- queries
    def Pivot(self):
        return self.Plan.GetPosition(0), self.Plan.GetDirection(0)

    def Speed(self):
        return float(np.ravel(Vector3.Length(self.Plan.GetVelocity(0)))[0])

    def Conditioning(self, root_position=None, root_forward=None, start=0):
        """traj_xz (45,2) [m] and traj_quat (45,4) wxyz for the model: plan samples start+1 .. start+45 relative to the
        pivot (sample `start` = the frame the chunk continues from, `start` frames ahead of now; `root_position`
        overrides the pivot point). The training data is yaw-augmented uniformly, so the frame is free: with
        `root_forward` the plan is expressed in the root's frame (+Z = hip forward), without it in world axes — the
        past-motion window must use the same frame (model_loop feeds world axes)."""
        assert 0 <= start <= self.Sim.SampleCount - 1 - FUTURE, start
        positions = np.asarray(Transform.GetPosition(self.Plan.Transforms), np.float64)[:, [0, 2]]
        directions = np.asarray(Transform.GetAxisZ(self.Plan.Transforms), np.float64)[:, [0, 2]]
        pivot = positions[start] if root_position is None else np.asarray(root_position, np.float64)[[0, 2]]
        xz = positions[start + 1:start + 1 + FUTURE] - pivot
        fwd = directions[start + 1:start + 1 + FUTURE]
        if root_forward is not None:
            yaw = yaw_of(np.asarray(root_forward, np.float64))
            xz, fwd = to_frame(xz, yaw), to_frame(fwd, yaw)
        quat = yaw_quat(np.arctan2(fwd[:, 0], fwd[:, 1]))
        return {"traj_xz": xz.astype(np.float32), "traj_quat": quat.astype(np.float32)}

    def Draw(self, sim=False):
        # One sample per frame is far too dense to read (76 spheres and 0.5 m arrows 3 cm apart at walking speed):
        # the path is a line through every sample, markers every DRAW_STRIDE frames (16 like the Biped demo), and the
        # LEAD_MAX extension past the 45-frame window the model actually sees is greyed out.
        Draw, Color = AI4Animation.Draw, AI4Animation.Color
        positions = Transform.GetPosition(self.Plan.Transforms)
        directions = Transform.GetAxisZ(self.Plan.Transforms)
        grey, pale = (170, 170, 170, 255), (255, 200, 140, 255)
        for lo, hi, colour in ((0, FUTURE + 1, Color.BLACK), (FUTURE, self.Plan.SampleCount, grey)):
            Draw.LineStrip(positions[lo:hi], color=colour)
        keys = np.arange(0, self.Plan.SampleCount, DRAW_STRIDE)
        for keys_, sphere, arrow in ((keys[keys <= FUTURE], Color.BLACK, Color.ORANGE), (keys[keys > FUTURE], grey, pale)):
            Draw.Sphere(positions[keys_], size=0.025, color=sphere)
            Draw.Cylinder(positions[keys_], positions[keys_] + DRAW_ARROW * directions[keys_], 0.02, 0.0, color=arrow)
        if sim and self.Prediction is not None:
            self.Sim.Draw(drawPositions=False, drawDirections=False, drawVelocities=False, positionColor=(160, 160, 160, 255))
