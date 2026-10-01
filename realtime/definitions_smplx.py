"""SMPL-X skeleton definitions for the realtime scene.

Canonical 55-joint order of the SMPL-X model (same order as smplx.joint_names.JOINT_NAMES[:55]
and the rows of J_regressor / columns of the npz kintree_table). The GLB built by smplx_glb.py
names its nodes with these strings, so the ai4animationpy Actor / SkinnedMesh resolve bones by
these names at runtime.
"""
import os
from pathlib import Path

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]
assert len(JOINT_NAMES) == 55

# Leg chain joint names — same roles as Geno's Definitions.py.
LeftHipName, LeftKneeName, LeftAnkleName, LeftBallName = "left_hip", "left_knee", "left_ankle", "left_foot"
RightHipName, RightKneeName, RightAnkleName, RightBallName = "right_hip", "right_knee", "right_ankle", "right_foot"

# flat_v1's 23 controlled joints (data/bodies.py REPO_NAMES) — the subset the persona
# model drives.
REPO_NAMES = [
    "pelvis", "left_hip", "left_knee", "left_ankle", "left_foot",
    "right_hip", "right_knee", "right_ankle", "right_foot",
    "spine1", "spine2", "spine3", "neck", "head", "jaw",
    "left_collar", "left_shoulder", "left_elbow", "left_wrist",
    "right_collar", "right_shoulder", "right_elbow", "right_wrist",
]

# SMPL-X neutral model (shapedirs / J_regressor / weights, 108 MB): $SMPLX_NPZ, else the repo's data/body_models
# (download it from https://smpl-x.is.tue.mpg.de and place it there; it is not redistributed with this code).
_SMPLX_CANDIDATES = [
    os.environ.get("SMPLX_NPZ", ""),
    str(Path(__file__).resolve().parents[1] / "data/body_models/smplx/SMPLX_NEUTRAL.npz"),
]
SMPLX_NPZ = next((p for p in _SMPLX_CANDIDATES if p and Path(p).exists()), _SMPLX_CANDIDATES[-1])
NUM_BETAS = 10
