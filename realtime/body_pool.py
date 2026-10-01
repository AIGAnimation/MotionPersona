"""The pool of real bodies the model was trained on, and how to sample from it.

Training-time shape augmentation never samples betas axis by
axis — it takes the 48 REAL subjects and mixes PAIRS of them convexly, which is exactly
the relation the MotionBuilder retargets already have in the data. Random shapes here
follow the same recipe, so what the sliders produce stays in-distribution.

data/bodies/bodies.npz also carries 80 'grid' bodies (a synthetic sweep); those are not
real subjects and are excluded from the random pool, but the model was trained on all
128 (the cross-body dataset), so the slider limits come from all of them.
"""
from pathlib import Path

import numpy as np

BODIES_NPZ = Path(__file__).resolve().parents[1] / "data/bodies/bodies.npz"


class BodyPool:
    def __init__(self, path=BODIES_NPZ, kind="real"):
        data = np.load(path, allow_pickle=True)
        keep = data["kind"] == kind
        self.Names = [str(n) for n in data["name"][keep]]
        self.Betas = np.asarray(data["betas"][keep], np.float64)
        self.Heights = np.asarray(data["height"][keep], np.float64)
        self.TrainingBetas = np.asarray(data["betas"], np.float64)      # every body the model has seen
        self.Rng = np.random.default_rng()

    def __len__(self):
        return len(self.Names)

    def Limits(self, step=None):
        """Per-axis (min, max) of the training bodies — the sliders must not leave them: the axes are far from
        symmetric (beta0 -6.9..2.6, beta9 -0.4..7.0, the rest within +-3.3), so a common +-8 was mostly out of
        distribution. `step` widens each bound outwards to the slider's snapping grid."""
        lo, hi = self.TrainingBetas.min(0), self.TrainingBetas.max(0)
        if step:
            lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
        return lo, hi

    def Random(self):
        """Convex mix of two random real bodies — the training augmentation's recipe.

        Returns (betas, label). The mixing weight is uniform, so pure subjects are as
        likely as any interpolation; the label names what was mixed.
        """
        i, j = self.Rng.choice(len(self.Names), size=2, replace=False)
        w = self.Rng.uniform()
        betas = (1.0 - w) * self.Betas[i] + w * self.Betas[j]
        label = (f"{self.Names[i]} {1 - w:.0%} / {self.Names[j]} {w:.0%}"
                 if 0.05 < w < 0.95 else self.Names[j if w > 0.5 else i])
        return betas, label
