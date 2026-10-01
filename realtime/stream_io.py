"""Stream export and scripted input for the realtime demo.

StreamRecorder -- what the player actually saw, in the rollout-archive layout (evaluation/archive.py), so a demo
session can be rendered and measured exactly like an evaluation rollout:

    <dir>/<key>.motion.npz  quats (T,23,4) wxyz local (root = world), root (T,3) y-up metres (y = 0 = the checkpoint's
                            basis plane: the sole plane for the released ckpt), contact (T,2), leg, body, and
                              betas (10,) + skel_offset (23,3)        when the body never changed, else
                              betas (T,10) + skel_offset (T,23,3)     the body each frame was GENERATED for
    <dir>/<key>.traj.npz    the eval fields -- cmd_xz_m, cmd_speed_mps, cmd_heading_rad, cmd_facing_rad, root_xz_m,
                            root_y_m, root_speed_mps, root_facing_rad, leg, speed_scale, v_cap_leg, block_ms,
                            frame_of_block -- plus the demo's own (see Export). cmd_* = the controller's own smoothed
                            command; cmd_xz_m is anchored to the root 0.5 s back (there is no world-fixed route)
    <dir>/events.json       header (checkpoint, settings, start state) + every input event + every block; the
                            `events` list is itself a valid --script, so a live take can be replayed

"Played stream" = the committed frames the playhead has passed: a committed frame ahead of the playhead can still be
dropped by ChunkStreamer.Invalidate, one behind it never. The recorder is display-rate independent (it collects whole
30 fps frames), holds the raw model output (no foot lock, no ground tracking -- like the archives) and starts at the
first generated frame (the rest-pose history the stream was reset with is not part of it, like the archives' canonical
history). Each frame carries the conditions of the chunk that produced it (subject, style, betas), so a body morph
or a persona switch lands on the right frames.

InputScript -- a timed list of inputs (the same `events` schema) that drives Program instead of the keyboard / pad:

    {"t": 0.0, "stick": [0, -1], "run": false}    left stick / WASD, held until the next stick event; "ramp": s
                                                  glides to it (smoothstep, direction along the short arc; a
                                                  top-level "ramp" sets the default, 0 = a keyboard step). Screen
                                                  convention of the live input: +y = W = world -Z (away from the
                                                  default camera), +x = D = world +X. The rest pose faces +Z, so
                                                  [0, -1] (S) walks straight on without a turn-around, and the
                                                  character's right is world -X, i.e. [-1, 0] (A).
    {"t": 3.0, "facing": 90}                      lock the facing at a world yaw (deg, +Z = 0, atan2(x, z); "ramp" too);
    {"t": 3.0, "facing": "current"}               ... at the current facing (the F key); null = unlock
    {"t": 5.0, "subject": "p33"}                also "persona"; "subject_step": +1 / -1 = the , . keys
    {"t": 6.0, "style": "angry"}                  "style_step": +1 / -1 = the [ ] keys
    {"t": 10.0, "body": "p13", "over": 2.5}    morph to a pool body (data/bodies/bodies.npz name, or "own" = the
                                                  current subject's own body) over `over` s (smoothstep; 0 = jump)
    {"t": 10.0, "betas": [..10..], "over": 2.5}   ... or to explicit betas
    {"t": 12.0, "foot_lock": true} / {"ui": true} / {"hud": false} / {"reset": true}
    {"t": 15.9, "cfg_chunks": 0} / {"cfg": 0.75}  the history-CFG boost (--cfg-chunks / --cfg) of the subject / style
                                                  changes that FOLLOW (a running boost is not cut); absent = the flags
    {"t": 20.0, "end": true}                      stop (headless: export and return; window: export and close)

A script file is either that list or {"events": [...], ...} (an exported events.json works as is).
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FPS = 30
CMD_LAG = 15                   # frames: cmd_xz_m = root 0.5 s ago + the commanded displacement since (see Export)
CMD_SMOOTH = 8                 # frames (gaussian sigma) of root smoothing under that anchor
BODIES_NPZ = REPO / 'data/bodies/bodies.npz'
EVENT_KEYS = ('stick', 'run', 'facing', 'subject', 'persona', 'subject_step', 'style', 'style_step', 'body', 'betas',
              'foot_lock', 'ui', 'hud', 'reset', 'cfg', 'cfg_chunks', 'end')


def _yaw(v):
    """Planar direction (x, z) -> yaw (rad, +Z = 0), the eval / training convention."""
    return float(np.arctan2(v[0], v[1]))


def _carry_yaw(vx, vz, eps=1e-3):
    """Per-frame yaw of a planar vector, carrying the last defined value through zero vectors."""
    yaw = np.arctan2(vx, vz)
    good = np.hypot(vx, vz) > eps
    if not good.any():
        return np.zeros_like(yaw)
    idx = np.maximum.accumulate(np.where(good, np.arange(len(yaw)), 0))
    idx[:np.argmax(good)] = np.argmax(good)
    return yaw[idx]


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return [round(float(x), 5) for x in v.ravel()]
    if isinstance(v, (np.floating, float)):
        return round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


# ----------------------------------------------------------------------------------------------------- the recorder
class StreamRecorder:
    """Collects the played stream of a ChunkStreamer; see the module docstring. Call `Attach(streamer)` once,
    `Step(streamer, control)` every rendered frame after `Advance`, `Event(t, **fields)` on every input change,
    `Restart()` whenever the streamer is Reset, `Export(...)` at the end."""

    def __init__(self):
        self.Restart()

    def Attach(self, streamer):
        streamer.OnCommit = self._on_commit

    def Restart(self, t0=0.0):
        """A new take: event times are counted from `t0` (the clock at the restart), so the take replays from 0."""
        self.T0 = float(t0)
        self.Quats, self.Root, self.Contact, self.Block, self.Control = [], [], [], [], []
        self.Blocks = {}
        self.Events = []
        self.Next = 0                        # next stream index to finalise
        self._control = None                 # latest per-render-frame control snapshot
        self._last_input = None

    def _on_commit(self, info):
        self.Blocks[int(info['block'])] = dict(info)

    @property
    def Frames(self):
        return len(self.Quats)

    def DisplayFrame(self):
        """Index (in the exported stream) of the frame on screen right now; 0 before the first one."""
        return max(len(self.Quats) - 1, 0)

    def Step(self, streamer, control):
        """Finalise every committed frame the playhead has reached; `control` = this render frame's input snapshot
        (dict), stored on the frames it finalises (the frame on screen when that input was active)."""
        self._control = control
        last = streamer.Base + int(np.floor(streamer.T))
        if self.Next < streamer.Base:                       # never happens when Step runs every frame (the streamer
            self.Next = streamer.Base                       # keeps K + 30 frames behind the playhead)
        while self.Next <= last:
            i = self.Next - streamer.Base
            if streamer.FrameBlock[i] >= 0:                 # the reset history is not part of the stream
                self.Quats.append(np.asarray(streamer.Quats[i], np.float32).copy())
                self.Root.append(np.asarray(streamer.Root[i], np.float32).copy())
                self.Contact.append(np.asarray(streamer.Contact[i], np.float32).copy())
                self.Block.append(int(streamer.FrameBlock[i]))
                self.Control.append(control)
            self.Next += 1

    def Event(self, t, **fields):
        ev = {'t': round(float(t) - self.T0, 4), 'frame': self.DisplayFrame()}
        ev.update({k: _jsonable(v) for k, v in fields.items()})
        self.Events.append(ev)
        return ev

    def Input(self, t, stick, run, facing):
        """Log the continuous input as events, only when it changes (stick to 0.05, facing to 1 degree)."""
        stick = np.round(np.asarray(stick, np.float64)[:2] / 0.05) * 0.05
        face = None if facing is None else round(float(np.degrees(_yaw(np.ravel(facing)[[0, 2]]))))
        cur = (tuple(np.round(stick, 2)), bool(run), face)
        if cur == self._last_input:
            return
        prev = self._last_input
        self._last_input = cur
        ev = {}
        if prev is None or prev[0] != cur[0] or prev[1] != cur[1]:
            ev.update(stick=[float(x) for x in cur[0]], run=cur[1])
        if prev is None or prev[2] != cur[2]:
            ev['facing'] = face
        self.Event(t, **ev)

    # ------------------------------------------------------------------------------------------------- export
    def Export(self, out_dir, key, model, header=None, bodies=None):
        """Write <key>.motion.npz / <key>.traj.npz / events.json into out_dir. `model` = the PersonaModel (skeleton
        offsets for the betas, root basis). Returns the motion path, or None when nothing was played yet."""
        from data.bodies import skeleton_stats                              # noqa: E402 (repo import, lazy)
        from model_loop import hip_forward
        T = len(self.Quats)
        if T == 0:
            return None
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        quats = np.stack(self.Quats).astype(np.float32)
        root = np.stack(self.Root).astype(np.float32)
        contact = np.stack(self.Contact).astype(np.float32)
        block = np.asarray(self.Block, np.int64)

        # the body each frame was generated for
        betas_t = np.stack([np.asarray(self.Blocks[b]['betas'], np.float32) for b in block])
        uniq, inv = np.unique(betas_t, axis=0, return_inverse=True)
        inv = np.ravel(inv)
        offs = np.stack([model.Offsets(b) for b in uniq]).astype(np.float32)
        legs = np.array([float(skeleton_stats(o[None].astype(np.float64))['leg'][0]) for o in offs])
        offs_t, leg_t = offs[inv], legs[inv]
        varying = len(uniq) > 1
        body_id = self._body_id(betas_t[0], bodies)
        motion = dict(quats=quats, root=root, contact=contact,
                      skel_offset=offs_t if varying else offs_t[0], betas=betas_t if varying else betas_t[0],
                      leg=float(leg_t[0]), body=body_id)
        if varying:
            motion['leg_t'] = leg_t.astype(np.float32)
        np.savez(out / f'{key}.motion.npz', **motion)

        # root path, facing (the data's hip-forward definition with each frame's own hips)
        root_xz = root[:, [0, 2]]
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(root_xz, axis=0), axis=-1) * FPS])
        fwd = np.stack([hip_forward(quats[i, 0], offs_t[i].astype(np.float64)) for i in range(T)])
        root_facing = np.arctan2(fwd[:, 0], fwd[:, 2])

        # The command. The demo has no world-fixed route: its plan is re-anchored at the actor every frame, so the
        # plain integral of the commanded velocity (kept as cmd_open_xz_m) drifts away from the character for good
        # -- along the path by the speed shortfall, sideways by every corner cut. cmd_xz_m is the anchored version:
        # where the character was CMD_LAG frames ago plus the displacement the controller has commanded since
        # (its own smoothed velocity, plan sample 0). It leads the character by ~0.5 s along the commanded
        # direction, bends first at a turn and meets the root path again within CMD_LAG frames.
        ctl = self.Control
        plan_v = np.array([c['plan_v'] for c in ctl], np.float64)             # (T,2) m/s world xz
        plan_dir = np.array([c['plan_dir'] for c in ctl], np.float64)         # (T,2)
        disp = np.concatenate([[[0.0, 0.0]], np.cumsum(plan_v[1:], 0) / FPS])  # commanded displacement since frame 0
        cmd_open = root_xz[0][None].astype(np.float64) + disp
        lag = np.maximum(np.arange(T) - CMD_LAG, 0)
        # anchored on the SMOOTHED root: the raw root sways ~2-3 cm per step and the band would wobble with it
        from scipy.ndimage import gaussian_filter1d
        anchor = gaussian_filter1d(root_xz.astype(np.float64), CMD_SMOOTH, axis=0, mode='nearest')
        cmd_xz = anchor[lag] + disp - disp[lag]
        in_v = np.array([c['input_v'] for c in ctl], np.float64)
        in_face = np.array([np.nan if c['input_facing'] is None else c['input_facing'] for c in ctl], np.float64)

        # blocks that put at least one frame on screen, in stream order
        order = [b for b in sorted(set(block.tolist()))]
        first = {b: int(np.argmax(block == b)) for b in order}
        kept = {b: int((block == b).sum()) for b in order}
        info = [self.Blocks[b] for b in order]
        tags = [self.Blocks[b].get('tag') or {} for b in block]
        traj = dict(
            cmd_xz_m=cmd_xz.astype(np.float32), cmd_open_xz_m=cmd_open.astype(np.float32),
            cmd_speed_mps=np.linalg.norm(plan_v, axis=1).astype(np.float32),
            cmd_heading_rad=_carry_yaw(plan_v[:, 0], plan_v[:, 1]).astype(np.float32),
            cmd_facing_rad=_carry_yaw(plan_dir[:, 0], plan_dir[:, 1]).astype(np.float32),
            root_xz_m=root_xz.astype(np.float32), root_y_m=root[:, 1].astype(np.float32),
            root_speed_mps=speed.astype(np.float32), root_facing_rad=root_facing.astype(np.float32),
            leg=float(leg_t[0]), speed_scale=1.0, v_cap_leg=-1.0,
            block_ms=np.array([i['ms'] for i in info], np.float32),
            frame_of_block=np.array([first[b] for b in order], np.int64),
            # --- demo extras
            block_id=np.array(order, np.int64), block_n=np.array([kept[b] for b in order], np.int64),
            block_hop=np.array([i['hop'] for i in info], np.int64),
            block_latency_ms=np.array([i['latency_ms'] for i in info], np.float32),
            block_cpu_ms=np.array([i.get('cpu_ms', np.nan) for i in info], np.float32),
            block_min_ms=np.array([i.get('min_ms', i['ms']) for i in info], np.float32),
            block_guidance=np.array([i['guidance'] for i in info], np.float32),
            frame_block=block,
            persona=np.array([str(t.get('subject', '')) for t in tags]),
            style=np.array([str(t.get('style', '')) for t in tags]),
            input_speed_mps=np.linalg.norm(in_v, axis=1).astype(np.float32),
            input_heading_rad=_carry_yaw(in_v[:, 0], in_v[:, 1]).astype(np.float32),
            input_facing_rad=in_face.astype(np.float32),
            input_stick=np.array([c['stick'] for c in ctl], np.float32),
            input_run=np.array([c['run'] for c in ctl], bool),
            ground_m=np.array([c.get('ground', 0.0) for c in ctl], np.float32),
            height_m=np.array([c.get('height', np.nan) for c in ctl], np.float32),
        )
        if varying:
            traj['leg_t'] = leg_t.astype(np.float32)
        np.savez(out / f'{key}.traj.npz', **traj)

        doc = dict(header or {})
        doc.update(version=1, fps=FPS, key=key, frames=T, body_varies=bool(varying),
                   files={'motion': f'{key}.motion.npz', 'traj': f'{key}.traj.npz'})
        doc['events'] = self.Events
        doc['blocks'] = [{'block': b, 'frame': first[b], 'n': kept[b], 'hop': int(self.Blocks[b]['hop']),
                          'ms': round(float(self.Blocks[b]['ms']), 2),
                          'cpu_ms': round(float(self.Blocks[b].get('cpu_ms', float('nan'))), 2),
                          'min_ms': round(float(self.Blocks[b].get('min_ms', self.Blocks[b]['ms'])), 2),
                          'latency_ms': round(float(self.Blocks[b]['latency_ms']), 2),
                          'guidance': float(self.Blocks[b]['guidance'])} for b in order]
        (out / 'events.json').write_text(json.dumps(doc, indent=1))
        return out / f'{key}.motion.npz'

    @staticmethod
    def _body_id(betas, bodies=None):
        """bodies.npz id of these betas (a pool body), or -1 (a custom / morphed body)."""
        if bodies is None:
            bodies = BodyTable()
        return bodies.Match(betas)


# ----------------------------------------------------------------------------------------------------- bodies
class BodyTable:
    """data/bodies/bodies.npz: the 128 training bodies by name / id."""

    def __init__(self, path=BODIES_NPZ):
        d = np.load(path, allow_pickle=True)
        self.Names = [str(n) for n in d['name']]
        self.Ids = [int(i) for i in d['id']]
        self.Betas = np.asarray(d['betas'], np.float64)
        self.Heights = np.asarray(d['height'], np.float64)

    def Get(self, name):
        if name not in self.Names:
            raise SystemExit(f'unknown body {name!r} (data/bodies/bodies.npz has {len(self.Names)})')
        return self.Betas[self.Names.index(name)].copy()

    def Match(self, betas, tol=1e-4):
        d = np.abs(self.Betas - np.asarray(betas, np.float64)[None]).max(1)
        i = int(np.argmin(d))
        return self.Ids[i] if d[i] < tol else -1

    def Name(self, betas, tol=1e-4):
        i = self.Match(betas, tol)
        return self.Names[self.Ids.index(i)] if i >= 0 else None


# ----------------------------------------------------------------------------------------------------- the script
def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def blend_stick(a, b, w):
    """Stick between a and b at weight w: magnitude linear, direction along the shorter arc (a zero end takes the
    other end's direction, so a start / stop only ramps the speed and a turn keeps the speed while it turns)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ma, mb = float(np.hypot(*a)), float(np.hypot(*b))
    if w >= 1.0:
        return b.copy()
    if ma < 1e-9 and mb < 1e-9:
        return np.zeros(2)
    ta = np.arctan2(a[1], a[0]) if ma >= 1e-9 else np.arctan2(b[1], b[0])
    tb = np.arctan2(b[1], b[0]) if mb >= 1e-9 else ta
    d = (tb - ta + np.pi) % (2 * np.pi) - np.pi
    th, m = ta + w * d, ma + w * (mb - ma)
    return m * np.array([np.cos(th), np.sin(th)])


class BodyMorph:
    """A running betas morph (smoothstep over `over` seconds): the script's `body ... over` events and the live
    B key (the prompted subject's own body) both drive one of these."""

    def __init__(self):
        self.State = None                    # (t0, t1, betas0, betas1, label)

    @property
    def Active(self):
        return self.State is not None

    def Start(self, t, betas0, betas1, over, label=None):
        self.State = None if over <= 0 else (float(t), float(t) + float(over), np.asarray(betas0, np.float64),
                                             np.asarray(betas1, np.float64), label)

    def Step(self, t):
        """-> (betas, done, label) at time t, or (None, True, None) without a morph."""
        if self.State is None:
            return None, True, None
        t0, t1, b0, b1, label = self.State
        w = smoothstep((t - t0) / (t1 - t0))
        done = t >= t1
        if done:
            self.State = None
        return (1.0 - w) * b0 + w * b1, done, label


class InputScript:
    """A timed input list (module docstring) replayed against Program's clock. `Due(t)` returns the discrete
    events whose time has come (each exactly once); the held state (stick, run, facing) and the running body morph
    are kept here and read every frame."""

    def __init__(self, source, bodies=None):
        if isinstance(source, (str, Path)):
            self.Path = str(source)
            doc = json.loads(Path(source).read_text())
        else:
            self.Path, doc = None, source
        events = doc['events'] if isinstance(doc, dict) else doc
        self.Header = doc if isinstance(doc, dict) else {}
        self.Events = sorted((dict(e) for e in events), key=lambda e: float(e['t']))
        for e in self.Events:
            unknown = set(e) - set(EVENT_KEYS) - {'t', 'frame', 'over', 'ramp', 'note'}
            if unknown:
                raise SystemExit(f'script event at t={e["t"]}: unknown keys {sorted(unknown)}')
        self.Bodies = bodies or BodyTable()
        self.Cursor = 0
        self.Stick = np.zeros(2)                 # the stick right now (StickAt)
        self.Run = False
        self.Ended = False
        # Stick / facing changes glide over `ramp` seconds (per event, else the script's top-level "ramp", else 0 =
        # a step, which is what a keyboard does): the direction turns along the short arc and the magnitude eases,
        # both with a smoothstep -- a thumb pushing a stick round, not a key swap. A step makes the controller's
        # velocity turn within ~0.1 s while its facing slews at 120 deg/s, i.e. the character crabs through the corner.
        self.Ramp = float(self.Header.get('ramp', 0.0)) if isinstance(self.Header, dict) else 0.0
        self._stick = (0.0, 0.0, np.zeros(2), np.zeros(2))      # (t0, duration, from, to)
        self._facing = None                                     # (t0, duration, yaw0, yaw1) of a ramped facing
        self._morph = BodyMorph()
        ends = [float(e['t']) for e in self.Events if e.get('end')]
        self.Duration = ends[0] if ends else (float(self.Events[-1]['t']) + 3.0 if self.Events else 0.0)

    def Due(self, t):
        out = []
        while self.Cursor < len(self.Events) and float(self.Events[self.Cursor]['t']) <= t + 1e-9:
            e = self.Events[self.Cursor]
            self.Cursor += 1
            if 'stick' in e:
                now = self.StickAt(t)
                self._stick = (float(e['t']), float(e.get('ramp', self.Ramp)), now,
                               np.asarray(e['stick'], np.float64)[:2])
            if 'run' in e:
                self.Run = bool(e['run'])
            if e.get('end'):
                self.Ended = True
            out.append(e)
        return out

    def StickAt(self, t):
        """The stick at time t: the last stick event, reached along a smoothstep arc over its ramp."""
        t0, dur, a, b = self._stick
        u = 1.0 if dur <= 0.0 else float(np.clip((t - t0) / dur, 0.0, 1.0))
        self.Stick = blend_stick(a, b, smoothstep(u))
        return self.Stick

    def StartFacing(self, t, yaw0, yaw1, ramp):
        """A locked-facing change to yaw1 (rad) that glides from yaw0 over `ramp` s (FacingAt)."""
        self._facing = (float(t), float(ramp), float(yaw0), float(yaw1))

    def FacingAt(self, t):
        """Yaw (rad) of a ramped facing change at time t, or None when none is running."""
        if self._facing is None:
            return None
        t0, dur, y0, y1 = self._facing
        u = 1.0 if dur <= 0.0 else float(np.clip((t - t0) / dur, 0.0, 1.0))
        d = (y1 - y0 + np.pi) % (2 * np.pi) - np.pi
        if u >= 1.0:
            self._facing = None
        return y0 + smoothstep(u) * d

    def Target(self, e, subject):
        """Betas an event morphs to ('body': name | 'own', or 'betas')."""
        if 'betas' in e:
            return np.asarray(e['betas'], np.float64)
        name = subject if e['body'] == 'own' else e['body']
        return self.Bodies.Get(name)

    def StartMorph(self, t, betas0, betas1, over):
        self._morph.Start(t, betas0, betas1, over)

    def MorphBetas(self, t):
        """-> (betas, done) of the running morph at time t, or (None, True) without one."""
        betas, done, _ = self._morph.Step(t)
        return betas, done
