"""Build a skinned, textured GLB of the SMPL-X body for the ai4animationpy renderer.

    .venv/bin/python smplx_glb.py [--betas 0 0 ...] [--texture assets/smplx_texture_f_alb.png]

Skeleton (55 joints, translation-only rest pose), LBS weights truncated to the top-4
influences per vertex, vertex normals from face accumulation, and the SMPL-X uv layout
(vt/ft) baked in by splitting vertices along uv seams (10475 -> ~11.3k verts). The albedo
is downscaled to <=2048 and embedded as JPEG. No pose correctives — plain LBS, matching
what SkinnedMesh does on the GPU.
"""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from pygltflib import (
    Accessor, Asset, Attributes, Buffer, BufferView, GLTF2,
    Image, Material, Mesh, Node, PbrMetallicRoughness, Primitive,
    Scene, Skin, Texture, TextureInfo,
)

from definitions_smplx import JOINT_NAMES, NUM_BETAS, SMPLX_NPZ

ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
FLOAT, UNSIGNED_SHORT = 5126, 5123
DEFAULT_TEXTURE = str(Path(__file__).parent / "assets/smplx_texture_f_alb.png")


def load_smplx(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {
        "v_template": np.asarray(d["v_template"], np.float64),          # (10475, 3)
        "shapedirs": np.asarray(d["shapedirs"], np.float64)[:, :, :NUM_BETAS],
        "J_regressor": np.asarray(d["J_regressor"], np.float64),        # (55, 10475)
        "weights": np.asarray(d["weights"], np.float64),                # (10475, 55)
        "faces": np.asarray(d["f"], np.int64),                          # (20908, 3)
        "faces_uv": np.asarray(d["ft"], np.int64),                      # (20908, 3) into vt
        "vt": np.asarray(d["vt"], np.float64),                          # (11313, 2)
        "parents": np.asarray(d["kintree_table"], np.int64)[0],         # (55,), [0] is bogus
    }


def shape_mesh(model, betas):
    v = model["v_template"] + model["shapedirs"] @ np.asarray(betas, np.float64)
    joints = model["J_regressor"] @ v
    return v, joints


def vertex_normals(v, faces):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    n = np.stack(
        [np.bincount(faces.ravel(), weights=np.repeat(fn[:, k], 3), minlength=len(v)) for k in range(3)],
        axis=1,
    )
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def uv_split(model):
    """Split vertices along uv seams: unique (vertex, uv) corner pairs become vertices.

    Returns (vert_map, uv, tris): vert_map (M,) indexes the ORIGINAL vertex arrays
    (positions/normals/skinning are gathered through it, so runtime morphs stay cheap),
    uv (M,2) in glTF convention (v flipped), tris (F,3) into the split list.
    """
    pairs = np.stack([model["faces"].ravel(), model["faces_uv"].ravel()], axis=1)
    uniq, inverse = np.unique(pairs, axis=0, return_inverse=True)
    vert_map = uniq[:, 0]
    uv = model["vt"][uniq[:, 1]].copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    tris = inverse.reshape(-1, 3)
    assert len(uniq) < 65536, f"{len(uniq)} split verts exceed uint16 indices"
    return vert_map, uv.astype(np.float32), tris.astype(np.int64)


def encode_texture(path, max_size=2048, quality=88):
    img = PILImage.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def top4_weights(weights):
    idx = np.argsort(-weights, axis=1)[:, :4]
    w = np.take_along_axis(weights, idx, axis=1)
    w = w / w.sum(axis=1, keepdims=True)
    return idx.astype(np.uint16), w.astype(np.float32)


def build_glb(model, betas, out_path, texture_path=None):
    v, joints = shape_mesh(model, betas)
    normals = vertex_normals(v, model["faces"])
    skin_idx, skin_w = top4_weights(model["weights"])
    vert_map, uv, tris = uv_split(model)
    parents = model["parents"].copy()
    parents[0] = -1

    blob = bytearray()
    views, accessors = [], []

    def add_view(data, target):
        while len(blob) % 4:
            blob.extend(b"\0")
        offset = len(blob)
        blob.extend(data)
        views.append(BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target))
        return len(views) - 1

    def add(data, target, ctype, atype, count, minmax=False):
        view = add_view(data.tobytes(), target)
        acc = Accessor(bufferView=view, componentType=ctype, count=count, type=atype)
        if minmax:
            acc.min = data.min(axis=0).tolist()
            acc.max = data.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    n_split = len(vert_map)
    a_pos = add(v[vert_map].astype(np.float32), ARRAY_BUFFER, FLOAT, "VEC3", n_split, minmax=True)
    a_nrm = add(normals[vert_map].astype(np.float32), ARRAY_BUFFER, FLOAT, "VEC3", n_split)
    a_uv = add(uv, ARRAY_BUFFER, FLOAT, "VEC2", n_split)
    a_jnt = add(skin_idx[vert_map], ARRAY_BUFFER, UNSIGNED_SHORT, "VEC4", n_split)
    a_wgt = add(skin_w[vert_map], ARRAY_BUFFER, FLOAT, "VEC4", n_split)
    a_idx = add(tris.astype(np.uint16).ravel(), ELEMENT_ARRAY_BUFFER, UNSIGNED_SHORT, "SCALAR", tris.size)

    # Inverse bind matrices: rest pose is translation-only, so IBM_i = translate(-J_i).
    # glTF matrices are column-major flattened: translation lives in elements 12..14.
    ibms = np.tile(np.eye(4, dtype=np.float32).ravel(), (55, 1))
    ibms[:, 12:15] = -joints.astype(np.float32)
    a_ibm = add(ibms, None, FLOAT, "MAT4", 55)

    images, textures, materials = [], [], []
    if texture_path and Path(texture_path).exists():
        img_view = add_view(encode_texture(texture_path), None)
        images.append(Image(bufferView=img_view, mimeType="image/jpeg"))
        textures.append(Texture(source=0))
        materials.append(Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture=TextureInfo(index=0), metallicFactor=0.0
            ),
            name="smplx_albedo",
        ))
    elif texture_path:
        print(f"warning: texture {texture_path} not found, building untextured")

    # The importer's FK walks nodes in index order and assumes parents precede children,
    # so the armature root MUST be node 0 and joints follow in kintree (parent-first) order.
    nodes = []
    for i, name in enumerate(JOINT_NAMES):
        local = joints[i] - (joints[parents[i]] if parents[i] >= 0 else 0.0)
        nodes.append(Node(name=name, translation=[float(x) for x in local],
                          children=[int(c) + 1 for c in np.where(parents == i)[0]] or None))
    mesh_node = len(nodes) + 1
    nodes.append(Node(name="smplx_mesh", mesh=0, skin=0))
    nodes.insert(0, Node(name="SMPLX", children=[1, mesh_node]))

    gltf = GLTF2(
        asset=Asset(version="2.0", generator="persona/realtime smplx_glb"),
        scene=0,
        scenes=[Scene(nodes=[0])],
        nodes=nodes,
        meshes=[Mesh(primitives=[Primitive(
            attributes=Attributes(POSITION=a_pos, NORMAL=a_nrm, TEXCOORD_0=a_uv,
                                  JOINTS_0=a_jnt, WEIGHTS_0=a_wgt),
            indices=a_idx,
            material=0 if materials else None,
        )])],
        skins=[Skin(inverseBindMatrices=a_ibm, joints=list(range(1, 56)), skeleton=1)],
        images=images,
        textures=textures,
        materials=materials,
        bufferViews=views,
        accessors=accessors,
        buffers=[Buffer(byteLength=len(blob))],
    )
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(out_path)
    print(f"wrote {out_path}: {n_split} split verts ({len(v)} source), {len(tris)} faces, "
          f"55 joints, textured={bool(materials)}, pelvis height {joints[0, 1] - v[:, 1].min():.3f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SMPLX_NPZ)
    ap.add_argument("--betas", type=float, nargs=NUM_BETAS, default=[0.0] * NUM_BETAS)
    ap.add_argument("--texture", default=DEFAULT_TEXTURE)
    ap.add_argument("--out", default=str(Path(__file__).parent / "assets/smplx_neutral.glb"))
    args = ap.parse_args()
    build_glb(load_smplx(args.model), args.betas, args.out, texture_path=args.texture)


if __name__ == "__main__":
    main()
