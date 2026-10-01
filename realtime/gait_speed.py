"""Gait speed setpoints: one uniform walk / run speed for every body and persona (default), or the shape- and
persona-dependent rule learned from the data (fitted offline into data/bodies/gait_speed.json).

Learned: for the live betas evaluate the per-class model that won the leave-one-body-out CV on the training data (on
the cross-body data the population cruise speed is linear in the leg length, because the retargeter scales speed with the
leg), then multiply by the persona factor of the prompted subject (the retargeter carries each source subject's own
pace onto every body: p02 runs well above the population, the children walk below it).  That is faithful to the data
but makes the controls depend on who is prompted — p10's run (0.5x) never leaves the model's walking regime — so
the default is uniform: the same two setpoints whoever / whatever body is on screen, and the model's own persona
shows in how it moves at that speed.  Sideways / backward movement scales the forward setpoint by the ratios measured
in the data in both modes.

    gait = GaitSpeed(uniform=(1.0, 1.9))  # walk / run [m/s] for everyone; uniform=None = learned rule
    gait.SetBody(betas)                  # -> gait.Walk, gait.Run  [m/s] (population rule), gait.Leg [m]
    gait.SetPersona('p02')                # -> gait.Persona = {'walk': f, 'run': f}; None = population
    v = gait.Setpoint(run=False, angle=90.0)   # walking sideways
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from data.bodies import betas_to_offsets, load_regressor, skeleton_stats  # noqa: E402

GAIT_SPEED_JSON = REPO / "data/bodies/gait_speed.json"
JOINT_REGRESSOR = REPO / "data/bodies/joint_regressor.npz"
G = 9.81


def features(kind, leg, betas, height=None):
    """Design row of the fitted speed model `kind` (data/bodies/gait_speed.json) for one body."""
    if kind == "const":
        return np.array([1.0])
    if kind == "lin_leg":
        return np.array([1.0, leg])
    if kind == "froude":
        return np.array([np.sqrt(G * leg)])
    if kind == "lin_sqrt_leg":
        return np.array([1.0, np.sqrt(leg)])
    if kind == "lin_height":
        if height is None:
            raise ValueError("lin_height model needs the mesh height")
        return np.array([1.0, height])
    if kind == "ridge_betas":
        return np.concatenate([[1.0], np.asarray(betas, np.float64)])
    raise KeyError(kind)


UNIFORM_WALK = 1.0   # m/s: the population walk at p02's leg length
UNIFORM_RUN = 1.9    # m/s: clearly in the model's running regime (it switches at 1.4-1.6 and tracks up to ~2.2)


class GaitSpeed:
    def __init__(self, path=GAIT_SPEED_JSON, regressor=JOINT_REGRESSOR, uniform=(UNIFORM_WALK, UNIFORM_RUN),
                 cap=None, ceiling=None):
        self.Spec = json.loads(Path(path).read_text())
        self.Classes = self.Spec["classes"]
        self.Regressor = load_regressor(regressor)
        self.Uniform = None if uniform is None else {"walk": float(uniform[0]), "run": float(uniform[1])}
        self.Adaptive = self.Uniform is None    # True = setpoints from the data (leg-length rule x persona factor)
        self.Pace = 1.0                 # user multiplier on top of the setpoint (slider)
        self.LegRate = None             # {'walk': legs/s, 'run': legs/s}: body-relative setpoints (--speed leg)
        self.Leg = self.Walk = self.Run = float("nan")
        self.Ratios = {gait: {"fwd": 1.0,
                              "side": float(self.Classes[f"{gait}/side"]["ratio_to_fwd"]),
                              "back": float(self.Classes[f"{gait}/back"]["ratio_to_fwd"])}
                       for gait in ("walk", "run")}
        self.Subjects = {s: r for s, r in self.Spec.get("persona", {}).items() if r.get("walk") and r.get("run")}
        # A persona may be at most as much faster than the population rule as the data's own p90 clip is above its
        # median (1.52 walk / 1.55 run). Four subjects sit above that -- p37, p43, p42, p08 -- because
        # the retargeter carried a child's cadence onto adult legs, so p37 on a 93 cm-leg body asked for
        # walk 2.25 / run 3.42 m/s (the data's run p90 over every clip is 1.91). Slow personas are not clamped:
        # p10 at 0.5x is real and harmless.
        self.FactorCap = {gait: float(self.Classes[f"{gait}/fwd"]["clip_speed_p10_50_90"][2]
                                      / self.Classes[f"{gait}/fwd"]["clip_speed_p10_50_90"][1])
                          for gait in ("walk", "run")} if cap is None else dict(cap)
        # ... and an absolute ceiling: no setpoint faster than the p90 clip of that gait over the whole dataset
        # (1.37 walk / 1.91 run m/s). The factor cap alone is relative to the body, so a child persona on a long
        # leg still asked for a 1.80 m/s "walk"; 1.91 is also about the uniform run setpoint (UNIFORM_RUN).
        self.Ceiling = {gait: float(self.Classes[f"{gait}/fwd"]["clip_speed_p10_50_90"][2])
                        for gait in ("walk", "run")} if ceiling is None else dict(ceiling)
        self.SetPersona(None)
        self.SetBody(np.zeros(10))

    def Model(self, cls):
        c = self.Classes[cls]
        return c["best"], np.asarray(c["models"][c["best"]]["coef"], np.float64)

    def Evaluate(self, cls, betas, height=None):
        betas = np.asarray(betas, np.float64)
        leg = float(skeleton_stats(betas_to_offsets(betas, self.Regressor))["leg"][0])
        kind, coef = self.Model(cls)
        return float(features(kind, leg, betas, height) @ coef), leg

    def SetBody(self, betas, height=None):
        """Re-evaluate the setpoints for a body (betas as on the sliders; height = mesh height if a model needs it)."""
        self.Betas = np.asarray(betas, np.float64).copy()
        self.Walk, self.Leg = self.Evaluate("walk/fwd", self.Betas, height)
        self.Run, _ = self.Evaluate("run/fwd", self.Betas, height)

    def SetAdaptive(self, on):
        """Runtime switch between the uniform setpoints and the data's own (the demo's Adaptive button / Y key).
        With `uniform=None` (--speed learned) there is nothing to switch back to, so it stays adaptive."""
        self.Adaptive = True if self.Uniform is None else bool(on)

    def SetPersona(self, subject):
        """Persona factor of the prompted source subject (None / unknown = population, factor 1)."""
        rec = self.Subjects.get(subject) if subject is not None else None
        self.Subject = subject if rec is not None else None
        self.Persona = {"walk": float(rec["walk"]), "run": float(rec["run"])} if rec is not None else {"walk": 1.0, "run": 1.0}

    def DirectionFactor(self, run, angle):
        """Speed ratio for moving at `angle` degrees (0..180) off the facing: fwd -> side (90) -> back (180) ratios."""
        r = self.Ratios["run" if run else "walk"]
        a = min(abs(float(angle)), 180.0)
        if a <= 90.0:
            return r["fwd"] + (r["side"] - r["fwd"]) * a / 90.0
        return r["side"] + (r["back"] - r["side"]) * (a - 90.0) / 90.0

    def Cruise(self, run=False):
        """Forward cruise speed [m/s]: the uniform setpoint, or body rule x persona factor; times the pace slider."""
        gait = "run" if run else "walk"
        if self.LegRate is not None:
            # body-relative command (--speed leg): the eval routes' convention, a fixed speed in leg lengths per
            # second on every body (walk v0 = 1.31 legs/s), no persona factor, no ceiling
            return self.Pace * self.LegRate[gait] * self.Leg
        base = self.Uniform[gait] if not self.Adaptive else self.Factor(gait) * (self.Run if run else self.Walk)
        return self.Pace * min(base, self.Ceiling[gait])      # pace stays a deliberate override above the ceiling

    def Factor(self, gait):
        """The prompted subject's speed factor, clamped to what the data's spread supports (see FactorCap)."""
        return min(self.Persona[gait], self.FactorCap[gait])

    def Label(self):
        """HUD tag for where the setpoints come from (a clamped factor is shown as 1.90>1.52)."""
        if self.LegRate is not None:
            return f"leg {self.LegRate['walk']:.2f} / {self.LegRate['run']:.2f} legs/s"
        if not self.Adaptive:
            return "uniform"
        if not self.Subject:
            return "population"
        parts = []
        for gait, run in (("walk", False), ("run", True)):
            raw, used = self.Persona[gait], self.Factor(gait)
            tag = f"x{raw:.2f}>{used:.2f}" if used < raw - 1e-9 else f"x{raw:.2f}"
            if used * (self.Run if run else self.Walk) > self.Ceiling[gait] + 1e-9:
                tag += " capped"
            parts.append(tag)
        return f"{self.Subject} {parts[0]} / {parts[1]}"

    def Setpoint(self, run=False, angle=0.0):
        return self.Cruise(run) * self.DirectionFactor(run, angle)

    @staticmethod
    def Mode(angle, moving=True):
        """HUD label for the movement direction relative to the facing."""
        if not moving:
            return "idle"
        a = abs(float(angle))
        return "forward" if a < 45.0 else ("strafe" if a < 135.0 else "backpedal")
