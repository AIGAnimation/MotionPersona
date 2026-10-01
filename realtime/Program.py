"""Persona realtime scene: the model in the loop (launched through ../realtime_demo.py).

    python realtime_demo.py                                   # default: the released prior
    python realtime_demo.py --subject p12 --style angry       # any subject x style x body
    python realtime_demo.py --model none                      # rest pose follows the plan

Features:
  - SMPL-X body rendered through the ai4animationpy pipeline (GLB from smplx_glb.py).
  - Locomotion control: WASD / gamepad drives the spring-damper trajectory controller
    (trajectory.py: RootModule.Series.Control laid out like the model's 45-frame traj
    conditioning, anchored to the actor root, data-matched sensitivities); the future
    trajectory is drawn live.
  - Facing decoupled from the movement (F / Q,E / right-mouse drag, gamepad right stick)
    so strafing and backpedalling are reachable, as in the data.
  - Speed setpoints (gait_speed.py): one uniform walk / run pair for every body and
    persona by default (--walk 1.0 / --run 1.9 m/s, so Shift always runs and the persona
    shows in the motion, not in the controls); --speed learned uses the data rule instead
    (leg-length population rule x the prompted subject's persona factor, p02 runs 1.4x the
    population, p10 0.5x). A Pace slider scales either.
  - 10 beta sliders morph the mesh + skeleton in place (see smplx_body.py).
  - --model: a trained head generates the motion chunk by chunk (model_loop.py): the
    last 10 frames + the plan -> the next 45 frames, requested in a background thread
    `--lead` frames before the buffer runs out, `--hop` frames committed per chunk (the
    rest is regenerated next time: small hop = responsive, 45 = strict chunk-by-chunk).
    Keys: [ ] style, , . subject (also the < > buttons of the Persona panel), V prompt
    variant (CLIP-prompt checkpoints), M model on/off, R restart the stream. Persona = the
    performer ID + typed attribute tokens (or the CLIP prompt of a text-token checkpoint),
    style = its own token, shape = the betas: any combination is allowed.
    Feel knobs: --sens (controller), --hop/--hop-steady/--lead/--blend (buffer dead
    time), --correction (Biped plan blend, off), --compile (CUDA graphs).
  - Foot lock (L, the Locomotion panel button, or --foot-lock to start on): the export
    path's runtime locking (utils/foot_lock.py) on the displayed pose only, so the
    model's own history stays raw. The HUD shows the skate it removes.

Recording the demo (formats in stream_io.py):
    --hide-ui / --ui       only the character + the two HUDs / the full panels (default); U toggles, H the HUDs
    --hud / --no-hud       input HUD (bottom-left: stick/WASD, persona, style, body height) + block-timing HUD
                           (top-right: last chunk ms vs the 100 ms budget, device); default = on with --hide-ui only
    --export DIR           write the played stream on exit (and on X): <key>.motion.npz + <key>.traj.npz + events.json,
                           the rollout-archive layout (evaluation/archive.py)
    --script EVENTS.json   drive the input from a timed list instead of the keyboard / pad (an exported events.json
                           replays as is)
    --headless             no window: script + model + controller on a fixed virtual clock, every chunk arriving one
                           render frame after its launch (reproducible with --seed); exports and exits at the end
    --capture OUT.mp4      window + fixed 30 fps virtual clock, every frame piped to ffmpeg (an automatic screen
                           recording of a script; slower than realtime, never drops a frame)
    --speed leg            walk / run at --walk-leg / --run-leg leg lengths per second (the eval routes' body-relative
                           command); --prompt-hop N, --timing-repeat N: see --help
    Keys: U panels, H HUDs, B morph to the prompted persona's own body, X keep this take (<export>/takeNN),
    R restart the stream (and the take)

Not implemented: inertial blending across chunk seams, timescale sync.
"""
import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import textwrap
from collections import deque
from pathlib import Path

# Same forward pipeline everywhere (macOS has no choice; on Linux the clone would pick the deferred one, whose
# bloom shader does not even link under Mesa). Must be set before ai4animation is imported.
os.environ.setdefault("AI4ANIMATION_PIPELINE", "forward")
if Path("/dev/dxg").exists():
    # WSL2: Mesa picks the llvmpipe software renderer unless the D3D12 (GPU paravirtualisation) driver is
    # requested explicitly; DISPLAY is WSLg's X server, unset in ssh sessions.
    os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
    os.environ.setdefault("DISPLAY", ":0")

import numpy as np  # noqa: E402
import raylib as rl  # noqa: E402

sys.path.append(str(Path(__file__).parent))

from ai4animation import (
    Actor,
    AI4Animation,
    Rotation,
    Time,
    Transform,
    Vector3,
)
from body_pool import BodyPool
from definitions_smplx import JOINT_NAMES, NUM_BETAS, REPO_NAMES, SMPLX_NPZ
from gait_speed import UNIFORM_RUN, UNIFORM_WALK, GaitSpeed
from smplx_body import SMPLXBody
from stream_io import BodyMorph, BodyTable, InputScript, StreamRecorder
from trajectory import LEAD_MAX, MOVE_SENSITIVITY, TRAJECTORY_CORRECTION, TURN_RATE, TrajectoryController

GLB_PATH = str(Path(__file__).parent / "assets/smplx_neutral.glb")
DEFAULT_MODEL = "checkpoints/prior.ckpt"  # the released prior (repo-relative; its codec sits next to it)
DRIVEN = [JOINT_NAMES.index(n) for n in REPO_NAMES]  # SMPL-X index of every repo joint
KEY_TURN_RATE = 120.0  # deg/s for the Q/E facing keys (TURN_RATE, imported from trajectory, is the plan's slew limit)
PACE_RANGE = (0.5, 1.5)  # user multiplier on the speed setpoints
# Slider limits = per-axis min/max of the 128 training bodies (BodyPool.Limits): anything
# beyond is out of distribution. Values snap to a coarse grid because the data never
# samples betas continuously (interpolation between bodies).
BETA_STEP = 0.2
UI_FONT = next((f for f in (
    "/System/Library/Fonts/Supplemental/Arial.ttf",          # macOS
    "/mnt/c/Windows/Fonts/arial.ttf",                        # WSL (the Windows font dir is mounted)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",       # Linux fallback
) if Path(f).exists()), None)
WINDOW_STATE = Path(__file__).parent / ".window.json"  # remembers size/position across runs
UI_TEXT = (95, 95, 95, 255)  # BVHView-style mid-gray labels
HUD_SIZE = 0.009  # status text height (fraction of the window height): half the panel labels
PROMPT_WRAP = 110  # characters per line of the prompt shown under the persona line (top centre)
# Adaptive hop: how much the control input has to move before the queued chunk counts as stale (m/s, and the chord
# between two unit facing targets: 0.09 ~ 5 degrees).
STEADY_SPEED = 0.15
STEADY_FACING = 0.09
# Ground tracking (see Program._ground): window of render frames whose lowest sole defines the floor, and how fast
# the correction may move so it never reads as bobbing.
GROUND_WINDOW = 90
GROUND_RATE = 0.25            # m/s
FOOT_JOINTS = [REPO_NAMES.index(n) for n in ("left_foot", "right_foot", "left_ankle", "right_ankle")]
# Row metrics of the framework's own left column (Camera.py: h 0.125, pitch 0.15 inside a 0.25-tall canvas), as
# fractions of the window height: keep the right column's buttons on the same vertical rhythm.
ROW_GAP = 0.00625
LOCO_H = 0.15                      # Locomotion panel: title pad + three rows at that gap
LOCO_ROW = 0.0342 / LOCO_H         # row height in canvas units (= the beta sliders', so labels match at 9 px)
LOCO_PITCH = (0.0342 + ROW_GAP) / LOCO_H
LOCO_TOP = 0.0209 / LOCO_H
# HUD anchors (fractions of the window width): clear of the framework's left column (x < 0.135) and of the right
# column (x > 0.845) while the panels are shown, in the corners when they are hidden.
HUD_LEFT = {True: 0.15, False: 0.02}
HUD_RIGHT = {True: 0.835, False: 0.98}
TIMING_WINDOW = 30                 # chunks averaged for the timing HUD's "avg"
OWN_BODY_MORPH = 0.6               # s: the B key morphs to the prompted subject's own body this fast


class Program:
    def __init__(self, args=None):
        self.Args = args if args is not None else parse_args([])

    def Start(self):
        self.Actor = AI4Animation.Scene.AddEntity("SMPLX").AddComponent(
            Actor, GLB_PATH, JOINT_NAMES, True
        )
        # The camera's Fixed/Third/Orbit modes assume the target sits on the GROUND and
        # add their own look-at height; targeting the pelvis-high actor entity would
        # aim over the head. A ground-level follower entity fixes the framing.
        self.CameraTarget = AI4Animation.Scene.AddEntity("CameraTarget")
        if AI4Animation.Standalone is not None:                # None = --headless (no window, no renderer)
            AI4Animation.Standalone.Camera.SetTarget(self.CameraTarget)

            # BVHView-style open horizon: drop the four dark backdrop walls, keep "Ground"
            # (routed to the checkerboard shader by the pipeline patch).
            pipeline = AI4Animation.Standalone.RenderPipeline
            for registered in list(pipeline.RegisteredModels):
                if registered.name.startswith("Wall"):
                    pipeline.UnregisterModel(registered.model)

        # Recording attachments: panels shown or not, the two HUDs, scripted input, the stream recorder.
        a = self.Args
        self.UIVisible = bool(a.ui)
        self.HUDOn = (not a.ui) if a.hud is None else bool(a.hud)
        self.Clock = 0.0                                   # seconds since the first Update (the script's time base)
        self.FixedDt = (1.0 / a.capture_fps) if a.capture else None
        self.Bodies = BodyTable()
        self.Morph = BodyMorph()                            # scripted `body ... over` events and the live B key
        self.Script = InputScript(a.script, self.Bodies) if a.script else None
        self.Recorder = StreamRecorder() if (a.export or a.headless) else None
        self.BlockMs = deque(maxlen=TIMING_WINDOW)
        self._chunks_seen = 0
        self._capture = None
        self._capture_frames = 0
        self._exported = False
        self.Done = False
        self._stick, self._run = [0.0, 0.0], False          # this frame's stick / run (HUD, recorder)
        self.DeviceLabel = ""
        if (a.headless or a.capture) and a.seed is None:
            a.seed = 0                                      # the offline drivers are reproducible by default

        self.Body = SMPLXBody(self.Actor, SMPLX_NPZ)
        self.Pool = BodyPool()
        self.ShapeLabel = "neutral"
        # start state: the command line, else the script's "start" block (an exported events.json carries one), else
        # p02 / neutral / the zero-beta body
        start = (self.Script.Header.get("start") or {}) if self.Script is not None else {}
        a.subject = a.subject or start.get("subject") or "p02"
        a.style = a.style or start.get("style") or "neutral"
        body = a.body or start.get("body")
        if body:
            self.Body.Apply(self.Bodies.Get(body))
            self.ShapeLabel = body
        elif start.get("betas") is not None and np.any(np.asarray(start["betas"], np.float64) != 0.0):
            self.Body.Apply(np.asarray(start["betas"], np.float64))
            self.ShapeLabel = "custom"
        cap, a = self.Args.persona_cap, self.Args
        pair = lambda w, r: None if w is None and r is None else {                       # noqa: E731
            "walk": (w if w is not None else float("inf")) or float("inf"),
            "run": (r if r is not None else float("inf")) or float("inf")}
        self.Gait = GaitSpeed(uniform=None if a.speed == "learned" else (a.walk, a.run),
                              cap=None if cap is None else {"walk": cap or float("inf"), "run": cap or float("inf")},
                              ceiling=pair(a.max_walk, a.max_run))
        if a.speed == "leg":
            self.Gait.LegRate = {"walk": float(a.walk_leg), "run": float(a.run_leg)}
        self.Gait.SetBody(self.Body.Betas)
        self.Trajectory = TrajectoryController(move=self.Args.sens, turn=self.Args.sens, correction=self.Args.correction,
                                               turn_rate=self.Args.turn_rate, steer=self.Args.steer)
        self.RootPosition = Vector3.Create(0.0, 0.0, 0.0)  # ground point under the actor root
        self.Facing = Vector3.Create(0.0, 0.0, 1.0)        # facing setpoint while locked
        self.FacingLock = False                            # False: face the movement direction
        self.Mode = "idle"
        self.Setpoint = 0.0
        self.DirectionMouseStart = None
        self.ControlInput = (np.zeros(3), np.zeros(3))     # (velocity setpoint, facing setpoint) of the last frame

        self.Model = None
        self.ModelActive = False
        rl.SetTargetFPS(int(self.Args.fps))
        if self.Args.model and self.Args.model.lower() != "none":
            self._start_model()

    # ------------------------------------------------------------------------------------------------ model loop
    def _start_model(self):
        from model_loop import ChunkStreamer, PersonaModel, TextFeats
        import torch
        a = self.Args
        if a.device == "auto":
            # cuda when there is one (in the loop it's the only option on a busy Linux desktop: the CPU path's
            # ~260 small ops per chunk fight the render loop for the GIL); otherwise cpu (mps stays opt-in).
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.Model = PersonaModel(a.model, device=a.device, steps=a.steps, ema=a.ema, threads=a.threads,
                                  compile=a.compile)
        self.Text = TextFeats()
        self.Subject = a.subject if a.subject in self.Text.Subjects else self.Text.Subjects[0]
        styles = self._styles()
        self.Style = a.style if a.style in styles else ("neutral" if "neutral" in styles else styles[0])
        self.Variant = 0
        self.Gait.SetPersona(self.Subject)
        self.Streamer = ChunkStreamer(self.Model, hop=a.hop, lead=a.lead, blend=a.blend, seed=a.seed,
                                      hop_steady=a.hop_steady, guidance=a.cfg, boost_chunks=a.cfg_chunks)
        self.Streamer.Reset(self.Body.Betas)
        if a.timing_repeat > 1:
            if not (a.headless or a.capture):
                raise SystemExit("--timing-repeat only with --headless / --capture (it would stall the live loop)")
            self.Streamer.TimingRepeat = int(a.timing_repeat)
        from hud import device_label
        self.DeviceLabel = device_label(a.device, torch.get_num_threads(),
                                        torch.cuda.get_device_name() if a.device == "cuda" else None)
        self._take_start()
        if self.Recorder is not None:
            self.Recorder.Attach(self.Streamer)
            if a.export:
                atexit.register(self._export)             # window closed / Ctrl-C / SIGTERM: the take is not lost
        self.ModelActive = True
        # Display-time foot lock (utils/foot_lock.py, the same runtime state machine as the export path): the
        # stream's own frames stay untouched, so the next chunk's history is still exactly what the model made.
        # `plant`: pull the locked toe down to the ground, which also removes the model's standing float (it holds
        # the lowest foot joint ~3.5 cm up, and the mesh sole hangs `self._sole()` below that joint, so the
        # character reads as floating ~2 cm). Ground is set at the sole, not the joint, so the mesh lands on y=0.
        from utils.foot_lock import FootLockStream
        self.FootLock = FootLockStream(self.Streamer.Offsets, self.Model.Parents, self.Model.Names,
                                       {'blend': a.lock_blend, 'plant': bool(a.plant)},
                                       ground=self._sole(), scale=100.0)          # config units are cm
        self.FootLockOn = bool(a.foot_lock)
        self.Ground = 0.0                                   # tracked floor offset, see _ground()
        self._GroundHist = deque(maxlen=GROUND_WINDOW)
        self._GroundResume = 0                              # hold the floor estimate until this chunk, see _reset_ground()
        self.Lift = float(a.lift)
        self.LaunchInput = self.ControlInput
        self.Relaunch = False
        s = self.Streamer
        print(f"model {self.Model.Name} ({self.Model.Kind}, {self.Model.Steps} steps, {a.device}"
              f"{f', compiled in {self.Model.CompileMs / 1000:.1f} s' if self.Model.CompileMs else ''}) | "
              f"hop {s.HopBase}-{s.HopSteady} lead {s.Lead} blend {s.Blend} | sens {a.sens} correction {a.correction} | "
              f"speed {self.Gait.Label()} walk {self.Gait.Cruise(False):.2f} run {self.Gait.Cruise(True):.2f} m/s")

    def _set_prompt(self, subject=None, style=None, variant=None):
        """Change the prompt (subject x style x phrasing); the next chunk request picks it up."""
        new_subject = subject is not None and subject != self.Subject
        if new_subject:
            self.Subject = subject
            self.Gait.SetPersona(subject)                     # the prompted subject's own pace
        styles = self._styles()
        if style is not None:
            self.Style = style
        if self.Style not in styles:
            self.Style = "neutral" if "neutral" in styles else styles[0]
        if variant is not None:
            self.Variant = variant % 3
        print(f"persona: {self.Subject} / {self.Style} v{self.Variant} | {self._persona_text()}")
        args = getattr(self, "Args", None)
        pb = int(getattr(args, "prompt_blend", 0) or 0)       # --prompt-blend N: cross-fade the change over N frames
        if pb > 0:
            self.Streamer.Transition(pb)
            self.Streamer.Invalidate(keep_tentative=True)
            self.Relaunch = True
        else:
            self._invalidate()                                # don't sit through a long hop of the old persona
        # ... and don't let the old style survive in the history. --style-cfg-chunks N: a change that keeps the
        # subject (style / phrasing) boosts N chunks instead of --cfg-chunks, without cutting short a subject
        # change's boost still in flight (the boost the persona switch needs also makes the feet glide under
        # angry / swimming; the style token takes on its own).
        style_chunks = getattr(args, "style_cfg_chunks", None)
        if new_subject or style_chunks is None:
            self.Streamer.Boost()
        else:
            self.Streamer.Boost(max(int(style_chunks), self.Streamer.BoostLeft))
        self._PromptHopLeft = int(getattr(args, "prompt_hop", 0) or 0)   # --prompt-hop: short hops while it takes
        if getattr(self, "Recorder", None) is not None:
            self.Recorder.Event(self.Clock, subject=self.Subject, style=self.Style)

    def _persona_text(self):
        """What the model is actually conditioned on, for the HUD: the CLIP prompt for a text-token checkpoint, the
        persona card (typed attributes) for one whose persona enters as learned tokens."""
        model = getattr(self, "Model", None)                  # None with --model none
        if model is None or model.PersonaCond == "text":
            return self.Text.Get(self.Subject, self.Style, self.Variant)[1]
        return f"ID {self.Subject}; {model.Rows.Describe(self.Subject)}"

    def _cycle_subject(self, step):
        subjects = self.Text.Subjects
        self._set_prompt(subject=subjects[(subjects.index(self.Subject) + step) % len(subjects)])

    def _styles(self):
        """The styles this checkpoint can actually be asked for (PersonaModel.AcceptsStyle); the prompt asset also
        lists label spellings that a learned style token has no row for."""
        model = getattr(self, "Model", None)                  # None with --model none
        styles = [s for s in self.Text.Styles(self.Subject) if model is None or model.AcceptsStyle(s)]
        return styles or self.Text.Styles(self.Subject)

    def _cycle_style(self, step):
        styles = self._styles()
        self._set_prompt(style=styles[(styles.index(self.Style) + step) % len(styles)])

    def _model_input(self):
        """Keyboard: style / subject / prompt variant cycling, model toggle, stream restart (the Persona panel's
        buttons do the same with the mouse, see GUI)."""
        if rl.IsKeyPressed(rl.KEY_M):
            self.ModelActive = not self.ModelActive
        if rl.IsKeyPressed(rl.KEY_R):
            self._restart_stream()
        if rl.IsKeyPressed(rl.KEY_X) and self.Recorder is not None and self.Args.export:
            self._export(take=True)                           # keep this take (<export>/takeNN), recording goes on
        if rl.IsKeyPressed(rl.KEY_L):
            self._set_foot_lock(not self.FootLockOn)
            if self.Recorder is not None:
                self.Recorder.Event(self.Clock, foot_lock=self.FootLockOn)
        if rl.IsKeyPressed(rl.KEY_Y):
            self._set_adaptive(not self.Gait.Adaptive)
        step = int(rl.IsKeyPressed(rl.KEY_RIGHT_BRACKET)) - int(rl.IsKeyPressed(rl.KEY_LEFT_BRACKET))
        if step:
            self._cycle_style(step)
        step = int(rl.IsKeyPressed(rl.KEY_PERIOD)) - int(rl.IsKeyPressed(rl.KEY_COMMA))
        if step:
            self._cycle_subject(step)
        if rl.IsKeyPressed(rl.KEY_V):
            self._set_prompt(variant=self.Variant + 1)
        if rl.IsKeyPressed(rl.KEY_B) and self.Subject in self.Bodies.Names:
            self._start_morph(self.Bodies.Get(self.Subject), OWN_BODY_MORPH, self.Subject)   # B: the persona's own body

    def _model_step(self):
        """Biped's Predict + Animate for a chunk model: top up the stream, play it back, pose the actor, and hand
        the stream's own future root path to the trajectory controller for next frame's correction blend."""
        from model_loop import assemble_smplx, fk
        s = self.Streamer
        s.Poll()
        if s.Error is not None:
            raise s.Error
        # Adaptive hop: standing still or walking a straight line, the plan the next chunk sees is the same plan, so
        # commit more of each chunk (a chunk every 400 ms instead of 100 ms — the model generates 45 frames either
        # way). The moment the input changes, throw away the dead time that bought us and go back to the short hop.
        changed = self._input_changed()
        if changed and s.Remaining() > s.Lead + s.HopBase:
            self._invalidate()
        # `Relaunch` matters: after a trim the buffer is deliberately above `Lead`, and waiting for it to drain would
        # put back the 100 ms the trim just saved.
        if not s.Busy() and (self.Relaunch or s.Remaining() <= s.Lead):
            start = min(s.Remaining(), LEAD_MAX)                      # the pivot is that many frames ahead of now
            cond = self.Trajectory.Conditioning(start=start)
            text = self.Model.Conditions(self.Subject, self.Style, self.Variant, texts=self.Text)   # CLIP row or learned rows
            steady = self._steady()
            if getattr(self, "_PromptHopLeft", 0) > 0:                # --prompt-hop N: a prompt change counts as an
                steady = False                                        # input change for the next N chunks
                self._PromptHopLeft -= 1
            s.Hop = s.HopSteady if steady else s.HopBase
            s.Launch(cond["traj_xz"], cond["traj_quat"], self.Body.Betas, text,
                     tag={"subject": self.Subject, "style": self.Style})
            if self.Args.headless or self.Args.capture:
                s.Wait()                  # offline drivers: compute the chunk now, alone (committed at the next Poll)
            self.LaunchInput = self.ControlInput
            self.Relaunch = False
        s.Advance(Time.DeltaTime)
        if s.Chunks > self._chunks_seen:                                  # a chunk was committed this frame
            self._chunks_seen = s.Chunks
            self.BlockMs.append(s.ComputeMs)
        quats, root, forward, _contact = s.Sample()
        if self.FootLockOn:
            self.FootLock.Ground = 100.0 * (self._sole() + self.Ground)   # the floor in the model's own frame (cm)
            quats = self.FootLock(quats, root, Time.DeltaTime).astype(np.float32)

        pos, rot = fk(quats[None], root[None].astype(np.float64), s.Offsets, self.Model.Parents)
        transforms = assemble_smplx(pos[0], rot[0], self.Body.Joints, self.Body.Parents, DRIVEN,
                                    self.Lift - self._ground(pos[0]))
        self.RootPosition = Vector3.Create(float(root[0]), 0.0, float(root[2]))
        self.RootForward = np.asarray(forward, np.float64).reshape(3)       # the body's facing (--facing-leash)
        root_tf = Transform.TR(self.RootPosition, Rotation.LookPlanar(forward.reshape(1, 3))[0])
        self.Actor.Root = root_tf
        self.Actor.Entity.SetTransform(root_tf)
        self.CameraTarget.SetPosition(self.RootPosition)
        self.Actor.SetTransforms(transforms)
        self.Actor.SyncToScene()

        path, fwd = s.FuturePath(self.Trajectory.Plan.SampleCount)
        path = path.copy()
        path[:, 1] = 0.0
        self.Trajectory.SetPrediction(path, fwd)
        if self.Recorder is not None:
            self.Recorder.Step(s, self._control_snapshot())

    def _control_snapshot(self):
        """What the controller was being asked for on this rendered frame (stream_io.StreamRecorder stores it on the
        frames it finalises): raw input, the plan's own velocity / facing (sample 0), display ground, body height."""
        vel, face = self.ControlInput
        plan_v = np.ravel(np.asarray(self.Trajectory.Plan.GetVelocity(0), np.float64))
        plan_dir = np.ravel(np.asarray(self.Trajectory.Pivot()[1], np.float64))
        return {"stick": [float(x) for x in self._stick], "run": bool(self._run),
                "input_v": [float(vel[0]), float(vel[2])],
                "input_facing": float(np.arctan2(face[0], face[2])) if float(np.linalg.norm(face)) > 0 else None,
                "plan_v": [float(plan_v[0]), float(plan_v[2])], "plan_dir": [float(plan_dir[0]), float(plan_dir[2])],
                "ground": float(self.Ground), "height": float(self.Body.Height)}

    def _ground(self, pos):
        """Where the model thinks the floor is, tracked from the motion itself [m].

        The model's absolute root height is only as good as the data's ground convention, and it varies with the
        subject and the speed: standing is ~2 cm of float, but e.g. `p21` walking at 1 m/s hovers ~6 cm (only
        41 % of frames put a foot below 5 cm) — which also starves the foot lock, whose contact test wants the toe
        under `height_thr`. So take the lowest sole of the last GROUND_WINDOW frames as the floor and drop the
        character onto it, rate-limited so the correction is never visible as bobbing."""
        if not self.Args.ground:
            return 0.0
        if self.Streamer.Chunks < self._GroundResume:
            # A body change is not instant for the stream: the frames still queued were generated for the OLD
            # skeleton, so FK with the new bones throws the character metres off (measured: 40 cm up one frame
            # after switching an adult for a 47 cm-leg child, and below the floor a moment later). Those frames
            # must not enter the window at all -- hold the estimate until the model has delivered chunks for the
            # new body, which `_reset_ground` already asked for by invalidating the queue.
            return self.Ground
        low = float(np.min(pos[FOOT_JOINTS, 1])) - self._sole()
        self._GroundHist.append(low)
        target = min(self._GroundHist)
        step = GROUND_RATE * max(float(Time.DeltaTime), 1e-4)
        self.Ground += float(np.clip(target - self.Ground, -step, step))
        return self.Ground

    def _reset_ground(self, scale):
        """Drop the character straight onto the floor on the next frame instead of creeping there at GROUND_RATE.

        A shape change moves the sole by whole centimetres at once (a 47 cm-leg child vs a 93 cm-leg adult), and
        both halves of the estimate are stale: the window still holds the old skeleton's heights, and the rate
        limit — which exists to hide the model's own drift — would take seconds to walk the offset over. So the
        estimate is scaled by `scale` = leg_new / leg_old (the hover is essentially the model's root-height
        convention, which scales with the skeleton) and the window is dropped, so nothing of the old body -- or of
        the transient while its queued frames drain -- survives into the new one."""
        self.Ground *= float(scale)
        self._GroundHist.clear()
        self._GroundResume = self.Streamer.Chunks + 2     # ... resume once the stream is on the new skeleton

    def _sole(self):
        """How far the mesh sole hangs below the model's own y=0 [m]: the model's root height is 'pelvis above the
        lowest foot JOINT' (joint-basis ckpts, ~3.5 cm) or 'pelvis above the rigid-foot sole plane' (sole-basis
        ckpts such as the released one, ~0.5-1 cm) -- Model.RootBasis picks it -- while what the eye judges
        is the mesh, whose pelvis height is measured from the mesh sole."""
        return float(self.Body.PelvisHeight) - float(self.Streamer.Offsets[0, 1])

    def _foot_lock_line(self):
        """HUD: the lock's state and what it is actually doing — mean toe travel while the feet are planted,
        before and after the IK (cm per 30 fps frame, the same metric as the export path's skate)."""
        if self.Model is None:
            return "Foot lock: n/a"
        if not self.FootLockOn:
            return "Foot lock: off (L)"
        raw, locked = self.FootLock.Skate()
        feet = sum(int(s["locked"]) for s in self.FootLock.State.values())
        return f"Foot lock: on   planted {feet}/2   skate {raw:.2f} -> {locked:.2f} cm/f"

    def _set_adaptive(self, on):
        """Y key / the Locomotion panel toggle: walk & run setpoints from the data for the prompted subject, scaled
        to this body's leg length (gait_speed.py), instead of the uniform 1.0 / 1.9 m/s pair."""
        self.Gait.SetAdaptive(on)
        if getattr(self, "AdaptiveButton", None) is not None:          # no panels without a window
            self.AdaptiveButton.Active = self.Gait.Adaptive

    def _set_foot_lock(self, on):
        """L key / the Locomotion panel toggle. Turning it on restarts the state machine (and its skate counters)
        so the feet do not snap to a stale anchor."""
        self.FootLockOn = bool(on)
        if self.FootLockOn:
            self.FootLock.SetBody(self.Streamer.Offsets)
        if getattr(self, "FootLockButton", None) is not None:
            self.FootLockButton.Active = self.FootLockOn

    def _invalidate(self):
        """The queued chunk no longer matches what the user is asking for (input, prompt or body changed): drop the
        dead time and request a fresh chunk on the next frame."""
        self.Streamer.Invalidate()
        self.Relaunch = True

    def _input_changed(self):
        """Has the user asked for something else since the chunk in flight was launched? (speed setpoint or facing)"""
        vel, face = self.ControlInput
        launched_vel, launched_face = self.LaunchInput
        return (float(np.linalg.norm(vel - launched_vel)) > STEADY_SPEED
                or float(np.linalg.norm(face - launched_face)) > STEADY_FACING)

    def _steady(self):
        """May the next chunk commit a long hop? Only if the plan the model will see is the plan it already saw:
        the input has not changed AND the character has caught up with it. Constant input is not enough — through
        the acceleration ramp and the body's own turn the plan keeps moving, and a long hop there costs 0.4 s of
        response (measured: idle->run t50 1.09 s vs 0.67 s)."""
        if self._input_changed():
            return False
        if self.Morph.Active:                                                      # a body morph: every chunk must
            return False                                                           # see the betas of its moment
        if abs(self.Trajectory.Speed() - self.Setpoint) > STEADY_SPEED:            # still accelerating / braking
            return False
        vel, face = self.ControlInput
        target = face if float(np.linalg.norm(face)) > 0.0 else vel                # unlocked: the facing follows the movement
        if float(np.linalg.norm(target)) > 0.0:
            current = np.ravel(np.asarray(self.Trajectory.Pivot()[1], np.float64))[:3]
            current = current / (float(np.linalg.norm(current)) or 1.0)
            if float(np.linalg.norm(current - target / np.linalg.norm(target))) > STEADY_FACING:
                return False                                                       # still turning
            if self.Args.steady_body > 0.0 and self.Model is not None:
                # --steady-body: the PLAN's facing reaching the target is not the character having turned -- the body
                # lags it by 30-45 deg for most of a second at walking speed, and a long hop committed there plays
                # that lag out as a crab walk. Also require the body (the facing of the last committed frame, the
                # next chunk's pivot) to be within this many degrees of the target.
                from model_loop import hip_forward
                s = self.Streamer
                body = np.ravel(hip_forward(np.asarray(s.Quats[-1])[0], s.Offsets))
                t = target / np.linalg.norm(target)
                err = np.degrees(abs(np.arctan2(body[0], body[2]) - np.arctan2(t[0], t[2])))
                if min(err, 360.0 - err) > self.Args.steady_body:
                    return False
        return True

    def _facing_input(self):
        """-> (facing direction or None, run flag, left stick). Facing = twin-stick style: the right stick / right
        mouse drag / Q,E set a facing that stays locked (strafe / backpedal), F (right stick click) releases it."""
        io = AI4Animation.Standalone.IO
        if io.GamepadAvailable():
            left_stick = io.GetLeftStick()
            right_stick = io.GetRightStick()
            run = io.IsLeftStickPressed()
            if io.IsRightStickPressed():
                self.FacingLock = False
        else:
            wasd = io.GetWASDQE()
            left_stick = [wasd[0], wasd[2]]
            run = rl.IsKeyDown(rl.KEY_LEFT_SHIFT)
            if rl.IsKeyPressed(rl.KEY_F):
                self.FacingLock = not self.FacingLock
                if self.FacingLock:
                    self.Facing = self.Trajectory.Pivot()[1].copy()
            turn = float(rl.IsKeyDown(rl.KEY_E)) - float(rl.IsKeyDown(rl.KEY_Q))
            if turn != 0.0:
                yaw = np.deg2rad(KEY_TURN_RATE * Time.DeltaTime * turn)
                f = np.ravel(self.Facing)
                self.Facing = Vector3.Create(
                    f[0] * np.cos(yaw) + f[2] * np.sin(yaw), 0.0, -f[0] * np.sin(yaw) + f[2] * np.cos(yaw)
                )
                self.FacingLock = True
            if rl.IsMouseButtonDown(rl.MOUSE_BUTTON_RIGHT):
                pos = np.array(io.GetMousePositionOnScreen())
                if self.DirectionMouseStart is None:
                    self.DirectionMouseStart = pos
                else:
                    momentum = 0.01
                    self.DirectionMouseStart = (1 - momentum) * self.DirectionMouseStart + momentum * pos
                right_stick = [pos[0] - self.DirectionMouseStart[0], self.DirectionMouseStart[1] - pos[1]]
            else:
                self.DirectionMouseStart = None
                right_stick = [0, 0]

        stick = Vector3.Create(right_stick[0], 0, -right_stick[1])
        if float(np.ravel(Vector3.Length(stick))[0]) > 0.0:
            self.Facing = Vector3.Normalize(stick)
            self.FacingLock = True
        return (self.Facing if self.FacingLock else None), run, left_stick

    def _script_input(self):
        """_facing_input for a script (stream_io.InputScript): the stick (gliding over an event's ramp), run, and the
        facing lock that the script's `facing` events set (see _apply_script_event)."""
        yaw = self.Script.FacingAt(self.Clock)
        if yaw is not None:
            self.Facing = Vector3.Create(np.sin(yaw), 0.0, np.cos(yaw))
        stick = self.Script.StickAt(self.Clock)
        return (self.Facing if self.FacingLock else None), self.Script.Run, [float(x) for x in stick]

    def Control(self):
        facing, run, left_stick = self._script_input() if self.Script is not None else self._facing_input()
        self._stick, self._run = left_stick, run
        if self.Recorder is not None and self.Script is None:       # a script logs its own events (ramps and all)
            self.Recorder.Input(self.Clock, left_stick, run, facing)
        move = Vector3.ClampMagnitude(Vector3.Create(left_stick[0], 0, -left_stick[1]), 1.0)
        moving = float(np.ravel(Vector3.Length(move))[0]) > 0.0

        # speed setpoint: learned per body shape, scaled by the movement direction relative to the facing
        angle = 0.0
        if moving and facing is not None:
            angle = float(np.ravel(Vector3.SignedAngle(facing, Vector3.Normalize(move), Vector3.UnitY()))[0])
        self.Setpoint = self.Gait.Setpoint(run, angle) if moving else 0.0
        self.Mode = ("run " if run else "walk ") + self.Gait.Mode(angle, moving) if moving else "idle"

        velocity = self.Setpoint * move
        direction = facing if facing is not None else Vector3.Create(0.0, 0.0, 0.0)
        # What the plan is being driven with: the adaptive hop compares this against the values the chunk in flight
        # was launched with (a zero facing = "follow the movement", so both halves matter).
        self.ControlInput = (np.ravel(np.asarray(velocity, np.float64))[:3].copy(),
                             np.ravel(np.asarray(direction, np.float64))[:3].copy())
        self.Trajectory.Control(self.RootPosition, direction, velocity, Time.DeltaTime,
                                root_forward=getattr(self, "RootForward", None), facing_leash=self.Args.facing_leash)

    def Update(self):
        if self.FixedDt is not None:
            Time.DeltaTime = self.FixedDt                     # --capture: a fixed virtual clock, one frame per video frame
        if self.Script is not None:
            self._script_step()                               # the script replaces every live key and the pad
        else:
            self._ui_keys()
            if self.Model is not None:
                self._model_input()
        self._morph_step()
        self.Control()
        if self.Model is not None and self.ModelActive:
            self._model_step()
            if self.FixedDt is not None and self.Streamer.Busy():
                self.Streamer.Wait()                          # the chunk lands on the next frame, however slow we are
        else:
            self._rest_step()
        self.Clock += float(Time.DeltaTime)

    # ------------------------------------------------------------------------------------ recording attachments
    def _ui_keys(self):
        """U: panels on/off (--hide-ui at runtime), H: the two HUDs on/off."""
        if AI4Animation.Standalone is None:
            return
        if rl.IsKeyPressed(rl.KEY_U):
            self.UIVisible = not self.UIVisible
            if self.Recorder is not None:
                self.Recorder.Event(self.Clock, ui=self.UIVisible)
        if rl.IsKeyPressed(rl.KEY_H):
            self.HUDOn = not self.HUDOn
            if self.Recorder is not None:
                self.Recorder.Event(self.Clock, hud=self.HUDOn)

    def _script_step(self):
        """Apply the script events whose time has come, then advance a running body morph."""
        sc = self.Script
        for e in sc.Due(self.Clock):
            self._apply_script_event(e)
        if sc.Ended and not self.Done:
            self._finish()

    def _start_morph(self, target, over, label):
        """Morph the body to `target` betas over `over` seconds (0 = jump), logged for the recorder."""
        if over > 0.0:
            self.Morph.Start(self.Clock, self.Body.Betas, target, over, label)
            if self.Model is not None:
                self._invalidate()                            # drop a long hop's dead time once, at the start
            self.ShapeLabel = f"-> {label}"
        else:
            self._set_body(target, log=False)
            self.ShapeLabel = label
        if self.Recorder is not None:
            self.Recorder.Event(self.Clock, body=label, betas=target, over=over)

    def _morph_step(self):
        betas, done, label = self.Morph.Step(self.Clock)
        if betas is not None:
            # every step goes through the slider path, minus its queue invalidation: invalidating every frame would
            # drop every chunk in flight and stall the stream; `_steady` keeps the short hop while a morph runs, so
            # each chunk is generated for the betas of its own moment
            self._set_body(betas, invalidate=False, log=False)
            if done:
                self.ShapeLabel = self.Bodies.Name(self.Body.Betas) or label or "custom"

    def _apply_script_event(self, e):
        if self.Recorder is not None:                         # the script's own input events, verbatim (with ramps)
            ev = {k: e[k] for k in ("stick", "run", "facing") if k in e}
            if "stick" in ev or isinstance(ev.get("facing"), (int, float)):
                ev["ramp"] = float(e.get("ramp", self.Script.Ramp))      # explicit, so a replay glides the same
            if ev:
                self.Recorder.Event(self.Clock, **ev)
        if "facing" in e:
            f = e["facing"]
            if f is None:
                self.FacingLock = False
            elif f == "current":
                self.FacingLock, self.Facing = True, self.Trajectory.Pivot()[1].copy()
            else:
                yaw = np.radians(float(f))
                ramp = float(e.get("ramp", self.Script.Ramp))
                if ramp > 0.0:                                # glide from the facing the plan has now
                    cur = np.ravel(np.asarray(self.Trajectory.Pivot()[1], np.float64))
                    self.Script.StartFacing(self.Clock, float(np.arctan2(cur[0], cur[2])), yaw, ramp)
                self.FacingLock = True
                self.Facing = Vector3.Create(np.sin(yaw), 0.0, np.cos(yaw)) if ramp <= 0.0 else \
                    self.Trajectory.Pivot()[1].copy()
        if self.Model is not None and ("cfg" in e or "cfg_chunks" in e):
            # history-CFG boost of the prompt changes that follow (before this event's own subject / style change).
            # A script can turn it off for style changes: the boost that makes the persona switch take also makes
            # the feet glide under angry / swimming.
            if "cfg" in e:
                self.Streamer.Guidance = float(e["cfg"])
            if "cfg_chunks" in e:
                self.Streamer.BoostChunks = int(e["cfg_chunks"])
        if self.Model is not None:
            subject = e.get("subject", e.get("persona"))
            style = e.get("style")
            if subject is not None or style is not None:
                self._set_prompt(subject=subject, style=style)
                if style is not None and self.Style != style:
                    print(f"WARNING script t={e['t']}: style {style!r} is not one {self.Subject} performed "
                          f"(Program._styles); playing {self.Style!r}")
            if e.get("subject_step"):
                self._cycle_subject(int(e["subject_step"]))
            if e.get("style_step"):
                self._cycle_style(int(e["style_step"]))
        if "body" in e or "betas" in e:
            target = self.Script.Target(e, self.Subject if self.Model is not None else None)
            label = e.get("body", "custom")
            if label == "own" and self.Model is not None:
                label = self.Subject
            self._start_morph(target, float(e.get("over", 0.0)), label)
        if "foot_lock" in e and self.Model is not None:
            self._set_foot_lock(bool(e["foot_lock"]))
        if "ui" in e:
            self.UIVisible = bool(e["ui"])
        if "hud" in e:
            self.HUDOn = bool(e["hud"])
        if e.get("reset") and self.Model is not None:
            self._restart_stream()
        for k in ("foot_lock", "ui", "hud", "reset", "cfg", "cfg_chunks"):
            if k in e and self.Recorder is not None:
                self.Recorder.Event(self.Clock, **{k: e[k]})

    def _set_body(self, values, label=None, invalidate=True, log=True):
        """New betas for the mesh, the skeleton the stream is generated for, the gait setpoints, the foot lock and
        the floor estimate -- the one path for the sliders, the Random / Reset buttons and a script."""
        values = np.asarray(values, np.float64)
        if np.allclose(values, self.Body.Betas):
            return False
        leg_before = float(self.Gait.Leg)
        self.Body.Apply(values)
        self.Gait.SetBody(values)
        if label is not None:
            self.ShapeLabel = label
        if self.Model is not None:                # the stream keeps its history; the next chunk sees the new skeleton
            self.Streamer.Offsets = self.Model.Offsets(values)
            if invalidate:
                self._invalidate()                # ... and it should see it now, not after a long hop
            self.FootLock.SetBody(self.Streamer.Offsets, ground=self._sole())
            self._reset_ground(self.Gait.Leg / max(leg_before, 1e-6))   # land now, don't creep
        for slider, v in zip(getattr(self, "Sliders", None) or [], values):
            slider.SetValue(float(v))
        if log and self.Recorder is not None:
            self.Recorder.Event(self.Clock, betas=values, over=0.0)
        return True

    def _restart_stream(self):
        """R: restart the stream from the rest pose (and the recording with it: a new take)."""
        self.Streamer.Reset(self.Body.Betas)
        self.FootLock.Reset()
        self.Trajectory.ClearPrediction()
        self._take_start()
        if self.Recorder is not None:
            self.Recorder.Restart(self.Clock)
            if self.Script is None:
                self.Recorder.Input(self.Clock, self._stick, self._run, self.Facing if self.FacingLock else None)

    def _take_start(self):
        """Start state of the take being recorded (export key + the events.json `start` block a replay begins from)."""
        self._StartKey = self._start_key()
        self._StartState = {"subject": self.Subject, "style": self.Style, "betas": [float(b) for b in self.Body.Betas],
                            "body": self.Bodies.Name(self.Body.Betas)}

    def _start_key(self):
        """Archive-style case key of the start state: <subject>__<body id>__<style>__s<seed> (body -1 = custom)."""
        body = self.Bodies.Match(self.Body.Betas)
        return f"{self.Subject}__{body:03d}__{self.Style}__s{self.Args.seed if self.Args.seed is not None else 0}"

    def _export_header(self):
        import time as _time
        a, m, s = self.Args, self.Model, self.Streamer
        mode = "headless" if a.headless else ("capture" if a.capture else "window")
        return {
            "created": _time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode, "script": a.script,
            "model": {"ckpt": str(a.model), "name": m.Name, "kind": m.Kind, "steps": m.Steps, "root_basis": m.RootBasis,
                      "persona_cond": m.PersonaCond, "style_cond": m.StyleCond, "ema": bool(a.ema)},
            "device": {"device": m.Device, "label": self.DeviceLabel},
            "settings": {"hop": s.HopBase, "hop_steady": s.HopSteady, "lead": s.Lead, "blend": s.Blend, "seed": a.seed,
                         "cfg": a.cfg, "cfg_chunks": a.cfg_chunks, "style_cfg_chunks": a.style_cfg_chunks,
                         "prompt_hop": a.prompt_hop, "prompt_blend": a.prompt_blend, "steady_body": a.steady_body,
                         "facing_leash": a.facing_leash, "steer": bool(a.steer),
                         "timing_repeat": a.timing_repeat, "script_ramp": (self.Script.Ramp if self.Script else None),
                         "sens": a.sens,
                         "turn_rate": a.turn_rate,
                         "correction": a.correction, "speed": a.speed, "walk": a.walk, "run": a.run,
                         "walk_leg": a.walk_leg, "run_leg": a.run_leg,
                         "render_fps": (a.capture_fps if a.capture else (a.headless_fps if a.headless else a.fps)),
                         "fixed_clock": bool(a.headless or a.capture)},
            "start": dict(self._StartState),
            "args": self._replay_args(),
            "units": "motion.npz: metres, wxyz quats, y-up, 30 fps, y = 0 = the checkpoint's basis plane "
                     "(root_basis); betas/skel_offset per frame when the body changed; raw model output "
                     "(no foot lock, no ground tracking)",
        }

    def _replay_args(self):
        """The flags that shape the stream, as --script reads them back ("args"), so the exported events.json replays
        under the same controller / speed / sampling settings (the output and driver flags are left out)."""
        a, d = self.Args, parse_args([])
        keep = ("model", "steps", "ema", "hop", "hop_steady", "lead", "blend", "cfg", "cfg_chunks", "style_cfg_chunks",
                "prompt_hop", "prompt_blend",
                "steady_body", "facing_leash", "sens", "turn_rate", "correction", "speed", "walk", "run", "walk_leg",
                "run_leg", "persona_cap", "max_walk", "max_run", "seed", "steer")
        out = []
        for k in keep:
            v = getattr(a, k)
            if v == getattr(d, k) and k != "seed":
                continue
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                out += [flag] if v else []
            elif v is not None:
                out += [flag, str(v)]
        return out

    def _export(self, take=False):
        """Write the played stream (stream_io.StreamRecorder.Export). take=True: into <export>/takeNN, keep going."""
        if self.Recorder is None or self.Model is None or not self.Args.export:
            return None
        if not take and self._exported:
            return None
        out = Path(self.Args.export)
        if take:
            self._takes = getattr(self, "_takes", 0) + 1
            out = out / f"take{self._takes:02d}"
        path = self.Recorder.Export(out, self.Args.export_key or self._StartKey, self.Model,
                                    header=self._export_header(), bodies=self.Bodies)
        if not take:
            self._exported = True
        print(f"stream: {self.Recorder.Frames} frames, {len(self.Recorder.Events)} events -> {path}")
        return path

    def _finish(self):
        """End of a script (or of --headless): export, close the capture, close the window."""
        if self.Done:
            return
        self.Done = True
        if self.Recorder is not None:
            self.Recorder.Event(self.Clock, end=True)
        self._export()
        if self._capture is not None:
            self._capture.stdin.close()
            self._capture.wait()
            print(f"capture: {self._capture_frames} frames -> {self.Args.capture}")
            self._capture = None
        if AI4Animation.Standalone is not None:
            AI4Animation.Standalone.Exit()

    def _capture_frame(self):
        """--capture: this frame's pixels (after every draw call of the frame) into the ffmpeg pipe."""
        rl.rlDrawRenderBatchActive()                          # flush the batched 2D draws before reading back
        img = rl.LoadImageFromScreen()
        w, h = int(img.width), int(img.height)
        if self._capture is None:
            if self.Done:
                rl.UnloadImage(img)
                return
            out = Path(self.Args.capture)
            out.parent.mkdir(parents=True, exist_ok=True)
            self._capture = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                 "-r", str(self.Args.capture_fps), "-i", "-",
                 "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,"
                        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white",
                 "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)],
                stdin=subprocess.PIPE)
            self._capture_size = (w, h)
        if (w, h) == self._capture_size:
            self._capture.stdin.write(bytes(rl.ffi.buffer(img.data, w * h * 4)))
            self._capture_frames += 1
        rl.UnloadImage(img)

    def _rest_step(self):
        # No model: the body rigidly follows the plan pivot in its rest pose.
        self.Trajectory.ClearPrediction()
        self.RootPosition = self.Trajectory.Pivot()[0]
        root = Transform.TR(
            self.RootPosition + Vector3.Create(0.0, self.Body.PelvisHeight, 0.0),
            Transform.GetRotation(self.Trajectory.Plan.Transforms[0]),
        )
        self.Actor.Root = root
        self.Actor.Entity.SetTransform(root)
        self.CameraTarget.SetPosition(self.RootPosition)
        self.Actor.SetTransforms(Transform.TransformationFrom(self.Body.RestTransforms, root))
        self.Actor.SyncToScene()

    def _restore_window(self):
        self._window_state = None
        if self.Args.capture:                                 # a capture is always 1920x1080, whatever was saved
            rl.SetWindowSize(1920, 1080)
            rl.SetWindowPosition(0, 0)
            self._window_state = "capture"                    # ... and does not overwrite the saved state
            return
        try:
            state = json.loads(WINDOW_STATE.read_text())
            rl.SetWindowSize(int(state["w"]), int(state["h"]))
            rl.SetWindowPosition(int(state["x"]), int(state["y"]))
            self._window_state = state
        except Exception:
            pass  # first run / stale file — keep the default 1920x1080

    def _persist_window(self):
        if self._window_state == "capture":
            return
        pos = rl.GetWindowPosition()
        state = {"w": rl.GetScreenWidth(), "h": rl.GetScreenHeight(),
                 "x": int(pos.x), "y": int(pos.y)}
        if state != self._window_state:
            self._window_state = state
            try:
                WINDOW_STATE.write_text(json.dumps(state))
            except Exception:
                pass

    def _restyle_ui(self):
        try:
            if UI_FONT is None:
                raise FileNotFoundError("no TTF font found")
            AI4Animation.Draw.SetFont(UI_FONT, 64)
        except Exception as exc:  # missing font file etc. — pixel font still works
            print("SetFont failed:", exc)

        # Widgets are native raygui (clone's GUI.py patch), BVHView-style: keep the
        # default light theme, just set fixed-size label text.
        rl.GuiSetStyle(0, 16, 14)  # DEFAULT TEXT_SIZE (group box titles)
        rl.GuiSetStyle(0, 17, 1)   # DEFAULT TEXT_SPACING
        # Button labels go through Draw.Text with Slider's formula (0.42 x box height, mid-grey), so a button and
        # the slider next to it read at exactly the same size -- raygui's own label is stuck on the global size.
        AI4Animation.GUI.Button.LabelLikeSlider = True

    def Standalone(self):
        self._restore_window()
        self._restyle_ui()
        self._install_ui_switch()
        # Right column, grouped BVHView-style; the framework's Camera/Actor/Scene
        # group boxes occupy the left column (Camera starts at y = 0.01: the top edges line up).
        self.ShapeCanvas = AI4Animation.GUI.Canvas("Body Shape", 0.845, 0.01, 0.14, 0.57)
        lo, hi = self.Pool.Limits(BETA_STEP)
        self.Sliders = [
            AI4Animation.GUI.Slider(
                0.08, 0.035 + 0.085 * i, 0.84, 0.055, 0.0, float(lo[i]), float(hi[i]),
                label=f"beta{i}", canvas=self.ShapeCanvas,
            )
            for i in range(NUM_BETAS)
        ]
        for slider, value in zip(self.Sliders, self.Body.Betas):  # a --body / script start body
            slider.SetValue(float(value))
        row = 0.035 + 0.085 * NUM_BETAS                       # one row, two buttons
        self.ResetButton = AI4Animation.GUI.Button("Reset", 0.08, row, 0.40, 0.055, state=False, canvas=self.ShapeCanvas)
        self.RandomButton = AI4Animation.GUI.Button("Random", 0.52, row, 0.40, 0.055, state=False, canvas=self.ShapeCanvas)
        self.LocoCanvas = AI4Animation.GUI.Canvas("Locomotion", 0.845, 0.62, 0.14, LOCO_H)
        # three rows, same height and same gap as the left column's buttons (a row and the slider above it then
        # render their label at exactly the same size too)
        rows = [LOCO_TOP + i * LOCO_PITCH for i in range(3)]
        self.PaceSlider = AI4Animation.GUI.Slider(
            0.08, rows[0], 0.84, LOCO_ROW, 1.0, PACE_RANGE[0], PACE_RANGE[1], label="pace", canvas=self.LocoCanvas,
        )
        self.AdaptiveButton = AI4Animation.GUI.Button(
            "Adaptive", 0.08, rows[1], 0.84, LOCO_ROW, state=self.Gait.Adaptive, canvas=self.LocoCanvas,
        )
        self.FootLockButton = None
        if self.Model is not None:
            self.FootLockButton = AI4Animation.GUI.Button(
                "Foot Lock", 0.08, rows[2], 0.84, LOCO_ROW, state=self.FootLockOn, canvas=self.LocoCanvas,
            )
        # Prompt panel (model only): < > buttons cycle the subject / style like the , . and [ ] keys.
        self.PersonaCanvas = None
        if self.Model is not None:
            self.PersonaCanvas = AI4Animation.GUI.Canvas("Persona", 0.845, 0.785, 0.14, 0.08)
            self.PersonaButtons = {
                (what, step): AI4Animation.GUI.Button(
                    "<" if step < 0 else ">", 0.05 if step < 0 else 0.79, y, 0.16, 0.36,
                    state=False, toggle=False, canvas=self.PersonaCanvas,
                )
                for what, y in (("subject", 0.10), ("style", 0.54)) for step in (-1, 1)
            }

    def Draw(self):
        if not self.Args.no_plan:
            self.Trajectory.Draw()

    def GUI(self):
        self._persist_window()
        if self.UIVisible:
            self._panels_gui()
        if self.HUDOn:
            self._draw_huds()

    def _draw_huds(self):
        """Input HUD bottom-left, block-timing HUD top-right (hud.py), clear of the panels when they are shown."""
        import hud
        model = self.Model is not None
        hud.draw_input_hud(HUD_LEFT[self.UIVisible], self._stick, self._run, self.FacingLock,
                           self.Subject if model else "-", self.Style if model else "-", self.Body.Height,
                           self.Bodies.Name(self.Body.Betas))
        if model:
            ms = self.BlockMs[-1] if self.BlockMs else 0.0
            avg = float(np.median(self.BlockMs)) if self.BlockMs else 0.0      # median: the cold-start chunk is ~0.7 s
            hud.draw_timing_hud(HUD_RIGHT[self.UIVisible], ms, avg, self.DeviceLabel, self.Streamer.Hop)

    def _install_ui_switch(self):
        """--hide-ui / U: with the panels hidden, the framework's own left column (Camera / Actor / Scene boxes) and
        its FPS / Entities text go too, not just ours -- both are drawn by the clone's Standalone.Update, so wrap it
        (the forward pipeline only, which Program always selects). --capture reads the frame back after every draw."""
        st = AI4Animation.Standalone
        full_update, full_gui = st.Update, AI4Animation.__GUI__

        def gui():
            if self.UIVisible:
                full_gui()
            else:
                self.GUI()                                    # our HUDs only (GUI skips the panels when hidden)
            if self.Args.capture:
                self._capture_frame()

        def update():
            if self.UIVisible:
                return full_update()
            AI4Animation.__UPDATE__()
            rl.BeginDrawing()
            st.RenderPipeline.Render(lambda: AI4Animation.__DRAW__())
            AI4Animation.__GUI__()
            rl.EndDrawing()

        AI4Animation.__GUI__ = staticmethod(gui)
        st.Update = update

    def _panels_gui(self):
        self.ShapeCanvas.GUI()
        modified = False
        for slider in self.Sliders:
            slider.GUI()
            if slider.Modified:  # dragged by hand — no longer a pool body
                self.ShapeLabel = "custom"
                modified = True

        self.ResetButton.GUI()
        if self.ResetButton.Active:
            self.ResetButton.Active = False
            for slider in self.Sliders:
                slider.SetValue(0.0)
            self.ShapeLabel = "neutral"
            modified = True

        self.RandomButton.GUI()
        if self.RandomButton.Active:
            self.RandomButton.Active = False
            betas, self.ShapeLabel = self.Pool.Random()
            for slider, value in zip(self.Sliders, betas):
                slider.SetValue(value)
            modified = True

        if modified:
            values = []
            for slider in self.Sliders:
                snapped = round(slider.GetValue() / BETA_STEP) * BETA_STEP
                slider.SetValue(snapped)
                values.append(snapped)
            self._set_body(values)                    # no-op when the snapped values did not move

        self.LocoCanvas.GUI()
        self.PaceSlider.GUI()
        self.Gait.Pace = float(self.PaceSlider.GetValue())
        self.AdaptiveButton.GUI()
        if bool(self.AdaptiveButton.Active) != self.Gait.Adaptive:
            self._set_adaptive(self.AdaptiveButton.Active)
        if self.FootLockButton is not None:
            self.FootLockButton.GUI()
            if bool(self.FootLockButton.Active) != self.FootLockOn:
                self._set_foot_lock(self.FootLockButton.Active)

        if self.PersonaCanvas is not None:
            self.PersonaCanvas.GUI()
            for (what, step), button in self.PersonaButtons.items():
                button.GUI()
                if button.Active:
                    button.Active = False
                    (self._cycle_subject if what == "subject" else self._cycle_style)(step)
            AI4Animation.Draw.Text(f"subject   {self.Subject}", 0.26, 0.21, 0.012, UI_TEXT, canvas=self.PersonaCanvas)
            AI4Animation.Draw.Text(f"style   {self.Style}", 0.26, 0.65, 0.012, UI_TEXT, canvas=self.PersonaCanvas)

        # Status text: half the size of the panel labels, lines kept short enough for the column width; the
        # per-frame speed numbers sit at the very bottom, under the static key help.
        AI4Animation.Draw.Text(
            f"Body: {self.ShapeLabel}   leg {self.Gait.Leg * 100:.0f} cm", 0.86, 0.595, HUD_SIZE, UI_TEXT,
        )
        AI4Animation.Draw.Text(
            f"Gait: {self.Mode}   Facing: {'locked' if self.FacingLock else 'movement'}\n"
            f"{self._foot_lock_line()}\n"
            "WASD move   Shift run   F lock facing   L foot lock   Y adaptive\n"
            "Q/E turn   RMB drag facing   Gamepad L3 run, R3 unlock\n"
            f"walk {self.Gait.Cruise(False):.2f} / run {self.Gait.Cruise(True):.2f} ({self.Gait.Label()})   "
            f"now {self.Trajectory.Speed():.2f} / {self.Setpoint:.2f} m/s",
            0.86, 0.873, HUD_SIZE, UI_TEXT,
        )
        if self.Model is not None:
            s = self.Streamer
            # Who is on screen: top centre (the left column is the framework's Camera / Actor / Scene boxes).
            prompt = self._persona_text()
            AI4Animation.Draw.Text(
                f"Persona: {self.Subject} / {self.Style} (v{self.Variant})   [ ] style   , . subject   V variant\n"
                + "\n".join(textwrap.wrap(prompt, PROMPT_WRAP)[:3]),
                0.15, 0.02, 0.018, UI_TEXT,
            )
            AI4Animation.Draw.Text(
                f"Model: {self.Model.Name} ({self.Model.Kind}, {self.Model.Steps} steps)   "
                f"{'ON' if self.ModelActive else 'OFF (M)'}   hop {s.Hop} lead {s.Lead}   R restart\n"
                f"Chunks {s.Chunks}   inference {s.LatencyMs:.0f} ms   buffer {s.Remaining()} f   stalls {s.Stalls}"
                + (f"   cfg {s.Guidance:.2f} x{s.BoostLeft}" if s.Boosting() else ""),
                0.30, 0.94, 0.018, UI_TEXT,
            )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=DEFAULT_MODEL, help="Lightning checkpoint of a trained head (ddpm / fm / latfm), repo-relative; "
                   "'none' = no model, the rest pose follows the plan")
    p.add_argument("--steps", type=int, default=None, help="sampling steps (latfm/fm: euler steps, default 2; ddpm: all)")
    p.add_argument("--ema", action="store_true", help="use the EMA weights stored in the checkpoint")
    p.add_argument("--device", default="auto", help="auto (default: cuda if available, else cpu), cpu, cuda or mps; the heads are "
                   "tiny (17 ms/chunk on 2 CPU threads); mps is only ~10%% faster per chunk and jitters more in the loop")
    p.add_argument("--fps", type=int, default=60, help="render frame-rate cap (0 = unlimited): the cap's sleep is what gives "
                   "the inference thread the GIL")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads for inference (default 2: the heads are tiny, 1-4 is fastest)")
    p.add_argument("--hop", type=int, default=3, help="frames committed per 45-frame chunk (45 = strict chunk-by-chunk; the committed "
                   "frames ahead of the playhead are input dead time, so small = responsive)")
    p.add_argument("--hop-steady", type=int, default=12, help="frames committed per chunk while the input does not change "
                   "(the plan is the same plan, so one chunk every 400 ms instead of 100 ms); any input change drops the "
                   "dead time and returns to --hop. Set equal to --hop to disable")
    p.add_argument("--lead", type=int, default=3, help="request the next chunk when this many frames are left (must cover the inference time)")
    p.add_argument("--foot-lock", action="store_true", help="start with the display-time foot lock on (L toggles it, "
                   "as does the Locomotion panel button): utils/foot_lock.py's runtime state machine + leg IK per "
                   "rendered frame; the stream's own frames stay raw, so the model's history never sees the IK")
    p.add_argument("--no-ground", dest="ground", action="store_false", help="do not track the floor from the motion: "
                   "show the model's own root height, float and all (see Program._ground)")
    p.add_argument("--lock-blend", type=float, default=0.1, help="foot lock inertialisation time (s) for lock / unlock")
    p.add_argument("--no-plant", dest="plant", action="store_false", help="with the foot lock on, keep the model's own "
                   "foot height instead of pulling the planted foot down onto the floor")
    p.add_argument("--cfg", type=float, default=0.5, help="history guidance weight used for a few chunks after a prompt "
                   "change (1 = off, <1 loosens the past pose so the new style can take, 0 = ignore the history). "
                   "Training drops the history token with p=cond_mask_prob, which is what makes this a trained mode")
    p.add_argument("--cfg-chunks", type=int, default=8, help="how many chunks after a prompt change use --cfg (0 = never)")
    p.add_argument("--style-cfg-chunks", type=int, default=None, help="chunks of --cfg after a style change that keeps "
                   "the subject (default: --cfg-chunks, like a subject change). 0 = style changes without the boost: "
                   "the boost makes a persona switch take, but under angry / swimming it also makes the feet glide, "
                   "and the style token takes on its own")
    p.add_argument("--compile", action="store_true", help="torch.compile the denoiser + VAE at startup (bit-exact; CUDA graphs "
                   "on cuda: 9.7 -> 6.0 ms per chunk on an RTX 5080, ~6 s of warm-up; no gain on CPU, broken on macOS)")
    p.add_argument("--blend", type=int, default=2, help="crossfade frames between overlapping chunks (< hop)")
    p.add_argument("--steer", action="store_true", help="facing unlocked: the plan's velocity turns with its facing "
                   "(at --turn-rate, speed eased separately), so heading == facing and a turn is an arc; default off = "
                   "the velocity lerps straight to the stick while the facing slews, i.e. the plan crabs through corners")
    p.add_argument("--facing-leash", type=float, default=0.0, help="keep the plan's facing within this many degrees "
                   "of the character's own facing (0 = off, the default: the plan turns on its own at --turn-rate and "
                   "runs 30-45 deg ahead of the body in a walking turn, which the model answers with a crab walk)")
    p.add_argument("--steady-body", type=float, default=0.0, help="adaptive hop: besides the plan, the character's "
                   "own facing must be within this many degrees of the target before a long hop (0 = off, the default: "
                   "only the plan is checked, so hop 12 resumes while the body is still turning; 20 is a good value)")
    p.add_argument("--prompt-blend", type=int, default=0, help="after a subject / style change, keep the dropped frames "
                   "as the old continuation and cross-fade the new chunk into it over this many frames (smoothstep; "
                   "carried into the next chunk when one hop is shorter). 0 = off, the default: the change cuts straight "
                   "to the new chunk with no blend at all (a visible pop); 12 = one steady hop")
    p.add_argument("--prompt-hop", type=int, default=0, help="after a subject / style change, commit --hop (not "
                   "--hop-steady) frames for this many chunks, like after a stick change (0 = off, the default: the "
                   "queued dead time is dropped at once, then the adaptive hop goes straight back to --hop-steady "
                   "because the trajectory input did not change)")
    p.add_argument("--sens", type=float, default=MOVE_SENSITIVITY, help="controller move/turn sensitivity (1/time constant): 3 = GT statistics, "
                   "5 = smooth, 8 = default (arcade)")
    p.add_argument("--turn-rate", type=float, default=TURN_RATE, help="heading slew limit of the plan [deg/s]: a real "
                   "turn in the data is a straight ramp whose per-frame rate has p99 172 and max 547 deg/s, so "
                   "the default 120 sits at the data's window-mean p99. 0 = the stock exponential turn driven by --sens")
    p.add_argument("--correction", type=float, default=TRAJECTORY_CORRECTION, help="Biped trajectory-correction weight towards the model's "
                   "own continuation (0 = off; 0.25 adds ~0.1 s of input lag)")
    p.add_argument("--speed", choices=("uniform", "learned", "leg"), default="uniform",
                   help="walk/run setpoints: uniform = --walk/--run for every body and persona (default); learned = the data "
                        "rule (leg-length x prompted subject's persona factor, data/bodies/gait_speed.json); leg = "
                        "--walk-leg/--run-leg leg lengths per second on every body (the eval routes' body-relative command)")
    p.add_argument("--walk-leg", type=float, default=1.31, help="--speed leg walk [legs/s] (eval route v0 = 1.31)")
    p.add_argument("--run-leg", type=float, default=2.44, help="--speed leg run [legs/s] (2.44 = 1.9 m/s on p02's leg)")
    p.add_argument("--persona-cap", type=float, default=None, help="max speed factor a persona may add on top of the "
                   "leg-length rule in adaptive mode (default: the data's own p90/p50 spread, 1.52 walk / 1.55 run; "
                   "0 = no cap). Without it a child persona on a long-legged body asks for 3.4 m/s")
    p.add_argument("--max-walk", type=float, default=None, help="absolute ceiling on the walk setpoint [m/s] "
                   "(default: the dataset's p90 walk clip, 1.37; 0 = no ceiling). The pace slider still scales past it")
    p.add_argument("--max-run", type=float, default=None, help="absolute ceiling on the run setpoint [m/s] "
                   "(default: the dataset's p90 run clip, 1.91; 0 = no ceiling)")
    p.add_argument("--walk", type=float, default=UNIFORM_WALK, help="uniform walk speed [m/s]")
    p.add_argument("--run", type=float, default=UNIFORM_RUN, help="uniform run speed [m/s] (the model runs from ~1.5, tracks to ~2.2)")
    p.add_argument("--seed", type=int, default=None, help="fixed sampling seed")
    p.add_argument("--subject", default=None, help="performer ID (default p02, or the script's start)")
    p.add_argument("--style", default=None, help="style (default neutral, or the script's start)")
    p.add_argument("--body", default=None, help="start on this training body (data/bodies/bodies.npz name, e.g. p02, "
                   "p13, grid_h09_g07); default the zero-beta body, or the script's start")
    p.add_argument("--lift", type=float, default=0.0, help="vertical display offset of the skeleton [m]")
    # --- recording attachments; none of them changes the default behaviour
    p.add_argument("--hide-ui", dest="ui", action="store_false", help="hide every panel (the right column, the "
                   "framework's left column, the status text): only the character and the two HUDs; U toggles")
    p.add_argument("--ui", dest="ui", action="store_true", help="show the panels (default)")
    p.set_defaults(ui=True)
    p.add_argument("--hud", dest="hud", action="store_true", default=None, help="draw the input HUD (bottom-left) "
                   "and the block-timing HUD (top-right); default: on with --hide-ui, off with --ui; H toggles")
    p.add_argument("--no-hud", dest="hud", action="store_false")
    p.add_argument("--no-plan", action="store_true", help="do not draw the controller's plan on the ground")
    p.add_argument("--script", default=None, help="events.json (stream_io.py): timed stick / facing / subject / style / "
                   "body events instead of the keyboard and the pad; an exported events.json replays as is")
    p.add_argument("--export", default=None, help="directory: write the played stream there on exit (X = keep a take "
                   "in <dir>/takeNN): <key>.motion.npz + <key>.traj.npz + events.json, the rollout-archive layout")
    p.add_argument("--export-key", default=None, help="file key (default <subject>__<body id>__<style>__s<seed> of "
                   "the start state, the evaluation archives' key format)")
    p.add_argument("--headless", action="store_true", help="no window: run --script on a fixed virtual clock "
                   "(--headless-fps), every chunk committed one frame after its launch; needs --export")
    p.add_argument("--headless-fps", type=float, default=60.0, help="render rate of the virtual clock (the live "
                   "demo's --fps cap)")
    p.add_argument("--duration", type=float, default=None, help="seconds to run --headless without a script end")
    p.add_argument("--capture", default=None, help="mp4: record the window frame by frame through ffmpeg on a fixed "
                   "--capture-fps virtual clock (slower than realtime, no dropped frames); use with --script")
    p.add_argument("--capture-fps", type=float, default=30.0)
    p.add_argument("--timing-repeat", type=int, default=1, help="offline drivers: also time every chunk as the fastest "
                   "of N runs of the same chunk (generator rewound, so the stream is unchanged) -> block_min_ms in the "
                   "export; a cost that survives a busy machine")
    a = p.parse_args(argv)
    if a.script:
        # a script may carry the command-line flags it was made for ("args": ["--steer", "--turn-rate", "75", ...]);
        # they go first, so anything given on the actual command line still wins
        try:
            doc = json.loads(Path(a.script).read_text())
        except (OSError, ValueError):
            doc = None
        extra = doc.get("args") if isinstance(doc, dict) else None
        if extra:
            a = p.parse_args([str(x) for x in extra] + list(sys.argv[1:] if argv is None else argv))
    if a.headless and not a.export:
        p.error("--headless needs --export")
    if a.headless and not a.script and not a.duration:
        p.error("--headless needs --script or --duration")
    return a


def run_headless(program):
    """--headless: Program without a window (AI4Animation MANUAL mode: Start, then Update by hand) on a fixed
    virtual clock. Every chunk is waited for right after its launch, so it is committed on the next frame: the
    stream no longer depends on the machine's speed or load, and with a seed it is reproducible."""
    AI4Animation(program, mode=AI4Animation.Mode.MANUAL)
    a = program.Args
    dt = 1.0 / float(a.headless_fps)
    end = program.Script.Duration if program.Script is not None else float(a.duration)
    if a.duration:
        end = min(end, float(a.duration))
    while not program.Done and program.Clock < end - 1e-9:
        AI4Animation.Update(dt)
        if program.Model is not None and program.Streamer.Busy():
            program.Streamer.Wait()
    program._finish()


if __name__ == "__main__":
    # A plain `kill` is a clean quit: a signal death makes macOS ask "reopen windows?" (modal, before InitWindow)
    # on the next launch of any python GUI.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    args = parse_args()
    if args.headless:
        run_headless(Program(args))
    else:
        AI4Animation(Program(args))
