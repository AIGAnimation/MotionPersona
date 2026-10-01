"""Runtime SMPL-X body: live betas -> mesh + skeleton updates on an ai4animationpy Actor.

The Actor loads the GLB once (smplx_glb.py output); this class then morphs the SAME GPU
buffers in place when betas change:
  - vertices/normals via raylib UpdateMeshBuffer (mesh was uploaded with dynamic=True),
  - SkinnedMesh.BindMatrices (numpy, translation-only rest pose -> translate(-J_i)),
  - rest transforms of the 55 joints (translation offsets from the pelvis), which the
    Program pushes into the scene every frame.

Geometry is morphed on the ORIGINAL 10475 vertices, then gathered through the same
uv-seam split (uv_split) and uint16-index chunking the GLB/renderer used, so buffer
updates stay aligned with what lives on the GPU.
"""
import numpy as np
import raylib as rl
from ai4animation.Standalone.SkinnedMesh import _split_indexed_mesh_into_chunks

from smplx_glb import load_smplx, shape_mesh, uv_split, vertex_normals

MESH_BUFFER_POSITIONS = 0  # raylib vboId slots
MESH_BUFFER_NORMALS = 2


class SMPLXBody:
    def __init__(self, actor, npz_path):
        self.Actor = actor
        self.Model = load_smplx(npz_path)
        self.VertMap, _, tris = uv_split(self.Model)
        self.Chunks = [c[0] for c in _split_indexed_mesh_into_chunks(tris)]
        self.Betas = np.zeros(self.Model["shapedirs"].shape[-1])
        self.Parents = self.Model["parents"].copy()
        self.Parents[0] = -1
        self.Joints = None          # (55,3) rest joints (SMPL-X frame), set by Apply
        self.RestTransforms = None  # (55,4,4), pelvis-relative, set by Apply
        self.PelvisHeight = 0.0     # pelvis y with the lowest vertex on the floor
        self.Height = 0.0           # rest-mesh height (m), the bodies.csv 'height' definition
        self.Apply(self.Betas)

    def Apply(self, betas):
        self.Betas = np.asarray(betas, np.float64).copy()
        v, joints = shape_mesh(self.Model, self.Betas)
        v_split = v[self.VertMap]
        n_split = vertex_normals(v, self.Model["faces"])[self.VertMap]

        skinned = getattr(self.Actor, "SkinnedMesh", None)   # None without a window (Program --headless)
        if skinned is not None:
            for model, chunk in zip(skinned.Models, self.Chunks):
                mesh = model.meshes[0]
                for slot, data in ((MESH_BUFFER_POSITIONS, v_split[chunk]), (MESH_BUFFER_NORMALS, n_split[chunk])):
                    buf = np.ascontiguousarray(data, np.float32)
                    rl.UpdateMeshBuffer(mesh, slot, rl.ffi.from_buffer("float[]", buf), buf.nbytes, 0)
            bind = np.tile(np.eye(4, dtype=np.float32), (55, 1, 1))
            bind[:, :3, 3] = -joints.astype(np.float32)
            skinned.BindMatrices = bind

        rest = np.tile(np.eye(4), (55, 1, 1))
        rest[:, :3, 3] = joints - joints[0]
        self.Joints = joints
        self.RestTransforms = rest
        self.PelvisHeight = float(joints[0, 1] - v[:, 1].min())
        self.Height = float(v[:, 1].max() - v[:, 1].min())
