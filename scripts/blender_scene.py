"""Build a Blender scene from generated motion: animated SMPL-X bodies, floor, lights, camera and the commanded route.

    blender --python scripts/blender_scene.py -- outputs/p02__p02__happy__s0.npz --smplx /path/to/SMPLX_NEUTRAL.npz
    blender -b --python scripts/blender_scene.py -- outputs/*.npz --smplx /path/to/SMPLX_NEUTRAL.npz -o outputs/scene.blend

Runs inside Blender (4.x / 5.x, its bundled Python and numpy; nothing from this repository's requirements). Takes the
`.npz` files written by generate.py (or the `*.motion.npz` of a realtime-demo export): local joint rotations, root
positions and the body's betas. Without `-b` Blender opens with the scene ready to play; with `-o` the scene is saved
as a .blend that opens anywhere without the SMPL-X model or this script.

Each clip becomes an armature whose rest pose is the SMPL-X skeleton of its body (betas -> joints by the SMPL-X joint
regressor) and, with --smplx, the SMPL-X mesh of that body skinned to it (fingers and eyes follow the wrist / head,
hands open, no pose-corrective blend shapes). Without --smplx only the armature is built. Clips are placed side by
side (--spacing); the commanded route of generate.py is drawn on the floor in blue. Y-up metres in the data become
Blender's Z-up; 30 fps.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

NAMES = ['pelvis', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'right_hip', 'right_knee', 'right_ankle',
         'right_foot', 'spine1', 'spine2', 'spine3', 'neck', 'head', 'jaw', 'left_collar', 'left_shoulder',
         'left_elbow', 'left_wrist', 'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 11, 15, 16, 17, 11, 19, 20, 21]
SMPLX_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle',
    'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplhf',
    'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3', 'left_middle1', 'left_middle2', 'left_middle3',
    'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1',
    'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1', 'right_middle2',
    'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
    'right_thumb1', 'right_thumb2', 'right_thumb3']
Y_UP_TO_Z_UP = Matrix.Rotation(math.pi / 2, 4, 'X')


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    ap = argparse.ArgumentParser(prog='blender --python scripts/blender_scene.py --', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz', nargs='+', help='generate.py outputs (<stem>.npz) or realtime exports (*.motion.npz)')
    ap.add_argument('--smplx', help='SMPLX_NEUTRAL.npz (https://smpl-x.is.tue.mpg.de); without it: skeletons only')
    ap.add_argument('-o', '--out', help='save the scene as this .blend')
    ap.add_argument('--spacing', type=float, default=1.5, help='sideways distance between clips (m)')
    ap.add_argument('--no-route', action='store_true', help='do not draw the commanded route')
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------- body model
class SMPLX:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.v_template = np.asarray(z['v_template'], np.float64)
        self.shapedirs = np.asarray(z['shapedirs'], np.float64)[:, :, :10]
        self.J_regressor = np.asarray(z['J_regressor'], np.float64)
        self.faces = np.asarray(z['f'], np.int64)
        parents = np.asarray(z['kintree_table'][0], np.int64)
        weights = np.asarray(z['weights'], np.float64)
        # skinning weights of the 55 SMPL-X joints folded onto the 23 animated ones (fingers -> wrist, eyes -> head)
        ours = {n: i for i, n in enumerate(NAMES)}
        self.weights = np.zeros((len(weights), len(NAMES)))
        for j, name in enumerate(SMPLX_NAMES):
            a = j
            while SMPLX_NAMES[a] not in ours:
                a = int(parents[a])
            self.weights[:, ours[SMPLX_NAMES[a]]] += weights[:, j]

    def rest(self, betas):
        b = np.zeros(10)
        b[:min(10, len(betas))] = np.asarray(betas, np.float64)[:10]
        verts = self.v_template + self.shapedirs @ b
        joints = self.J_regressor @ verts
        return verts, joints[[SMPLX_NAMES.index(n) for n in NAMES]]


def rest_from_offsets(offsets):
    """Rest joints from the parent-relative offsets alone (no SMPL-X model): pelvis at its height above the floor."""
    off = np.asarray(offsets, np.float64)
    joints = np.zeros_like(off)
    joints[0] = [0.0, off[0, 1], 0.0]
    for j in range(1, len(off)):
        joints[j] = joints[PARENTS[j]] + off[j]
    return joints


def stick_figure(joints, radius=0.025, sides=8):
    """Renderable stand-in for the body without the SMPL-X model: one cylinder per bone segment (rigidly bound to the
    parent joint) and a sphere for the head. Returns (verts, faces, weights) like the SMPL-X body."""
    verts, faces, owner = [], [], []
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    for j, p in enumerate(PARENTS):
        if p < 0:
            continue
        a, b = joints[p], joints[j]
        d = b - a
        if np.linalg.norm(d) < 1e-4:
            continue
        d = d / np.linalg.norm(d)
        u = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0]); u /= np.linalg.norm(u)
        w = np.cross(d, u)
        ring = radius * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w)
        base = len(verts)
        verts += list(a + ring) + list(b + ring)
        faces += [(base + i, base + (i + 1) % sides, base + sides + (i + 1) % sides, base + sides + i) for i in range(sides)]
        owner += [p] * (2 * sides)
    h = NAMES.index('head')                                           # head: a coarse sphere above the head joint
    c = joints[h] + np.array([0.0, 0.08, 0.0])
    lat, lon = np.linspace(-np.pi / 2, np.pi / 2, 7), ang
    base = len(verts)
    for la in lat:
        verts += [c + 0.1 * np.array([np.cos(la) * np.cos(lo), np.sin(la), np.cos(la) * np.sin(lo)]) for lo in lon]
    for i in range(len(lat) - 1):
        faces += [(base + i * sides + k, base + i * sides + (k + 1) % sides, base + (i + 1) * sides + (k + 1) % sides,
                   base + (i + 1) * sides + k) for k in range(sides)]
    owner += [h] * (len(lat) * sides)
    weights = np.zeros((len(verts), len(NAMES)))
    weights[np.arange(len(verts)), owner] = 1.0
    return np.asarray(verts), faces, weights


# ------------------------------------------------------------------------------------------------- scene parts
def material(name, rgba, rough=0.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Roughness'].default_value = rough
    return m


def build_armature(name, joints):
    """Armature in data space (y-up); bones point to their first child so the rig reads in the viewport."""
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    children = {j: [c for c, p in enumerate(PARENTS) if p == j] for j in range(len(NAMES))}
    for j, n in enumerate(NAMES):
        b = arm.edit_bones.new(n)
        b.head = Vector(joints[j])
        if children[j]:
            tail = joints[children[j][0]] if n != 'pelvis' else joints[9]
        else:                                                         # leaves continue their parent's direction
            d = joints[j] - joints[PARENTS[j]]
            tail = joints[j] + 0.5 * d
        if np.linalg.norm(tail - joints[j]) < 1e-4:
            tail = joints[j] + np.array([0.0, 0.05, 0.0])
        b.tail = Vector(tail)
        b.roll = 0.0
        if PARENTS[j] >= 0:
            b.parent = arm.edit_bones[NAMES[PARENTS[j]]]
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.matrix_world = Y_UP_TO_Z_UP
    return obj


def build_mesh(name, verts, faces, weights, armature, mat):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts).tolist(), [], np.asarray(faces).tolist())
    me.update()
    me.shade_smooth() if hasattr(me, 'shade_smooth') else None
    me.materials.append(mat)
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    for j, n in enumerate(NAMES):
        g = obj.vertex_groups.new(name=n)
        idx = np.nonzero(weights[:, j] > 1e-6)[0]
        for w in np.unique(np.round(weights[idx, j], 4)):              # one add() per distinct weight value
            sel = idx[np.round(weights[idx, j], 4) == w]
            g.add(sel.tolist(), float(w), 'REPLACE')
    obj.parent = armature
    mod = obj.modifiers.new('Armature', 'ARMATURE')
    mod.object = armature
    return obj


def fcurve_keys(action, path, index, values, group):
    fc = action.fcurves.new(path, index=index, action_group=group) if hasattr(action, 'fcurves') else None
    if fc is None:
        return False
    fc.keyframe_points.add(len(values))
    co = np.empty(2 * len(values), np.float32)
    co[0::2] = np.arange(1, len(values) + 1)
    co[1::2] = values
    fc.keyframe_points.foreach_set('co', co)
    for k in fc.keyframe_points:
        k.interpolation = 'LINEAR'
    fc.update()
    return True


def animate(obj, quats, root, joints_rest):
    """SMPL-style local rotations (identity rest frames) -> pose-bone rotations P = B^-1 q B for rest orientation B."""
    T = len(quats)
    obj.animation_data_create()
    action = bpy.data.actions.new(obj.name + '_action')
    obj.animation_data.action = action
    pose = obj.pose
    rest = {b.name: b.bone.matrix_local.to_3x3() for b in pose.bones}
    keyed = []
    for j, n in enumerate(NAMES):
        pb = pose.bones[n]
        pb.rotation_mode = 'QUATERNION'
        B = rest[n].to_quaternion()
        Bi = B.inverted()
        P = np.empty((T, 4))
        for t in range(T):
            q = Quaternion(quats[t, j])                               # wxyz
            P[t] = tuple(Bi @ q @ B)
        keyed.append((pb, 'rotation_quaternion', P))
    pb = pose.bones['pelvis']
    loc = (root - joints_rest[0]) @ np.asarray(rest['pelvis']).astype(np.float64)   # armature -> bone-local axes
    keyed.append((pb, 'location', loc))
    fast = True
    for pb, attr, vals in keyed:
        path = f'pose.bones["{pb.name}"].{attr}'
        for i in range(vals.shape[1]):
            if not fcurve_keys(action, path, i, vals[:, i], pb.name):
                fast = False
                break
        if not fast:
            break
    if not fast:                                                      # layered actions (Blender 4.4+): keyframe_insert
        obj.animation_data.action = None
        for t in range(T):
            for pb, attr, vals in keyed:
                setattr(pb, attr, tuple(vals[t]))
                pb.keyframe_insert(attr, frame=t + 1, group=pb.name)


def build_route(name, xz, offset_x, mat):
    pts = [Vector((x + offset_x, -z, 0.003)) for x, z in xz[::3]]
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = 0.02
    sp = curve.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for p, v in zip(sp.points, pts):
        p.co = (v.x, v.y, v.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    obj.scale = (1.0, 1.0, 0.05)                                       # flat ribbon on the floor
    return obj


def build_stage(target, n_frames):
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = 'floor'
    m = bpy.data.materials.new('floor')
    m.use_nodes = True
    nt = m.node_tree
    checker = nt.nodes.new('ShaderNodeTexChecker')
    checker.inputs['Scale'].default_value = 200.0                     # 1 m squares on the 200 m plane
    checker.inputs['Color1'].default_value = (0.80, 0.80, 0.80, 1)
    checker.inputs['Color2'].default_value = (0.65, 0.65, 0.65, 1)
    nt.links.new(checker.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color'])
    floor.data.materials.append(m)

    sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(40), 0, math.radians(30))
    bpy.context.collection.objects.link(sun)

    # camera: follows the first character's pelvis (horizontal position only) from a 3/4 view
    pivot = bpy.data.objects.new('camera_pivot', None)
    bpy.context.collection.objects.link(pivot)
    c = pivot.constraints.new('COPY_LOCATION')
    c.target, c.subtarget, c.use_z = target, 'pelvis', False
    cam = bpy.data.objects.new('camera', bpy.data.cameras.new('camera'))
    bpy.context.collection.objects.link(cam)
    cam.parent = pivot
    cam.location = (3.5, -4.5, 1.6)
    look = bpy.data.objects.new('camera_target', None)
    bpy.context.collection.objects.link(look)
    look.parent = pivot
    look.location = (0, 0, 0.9)
    t = cam.constraints.new('TRACK_TO')
    t.target, t.track_axis, t.up_axis = look, 'TRACK_NEGATIVE_Z', 'UP_Y'

    scn = bpy.context.scene
    if scn.world is None:
        scn.world = bpy.data.worlds.new('world')
    scn.world.use_nodes = True
    bg = scn.world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.85, 0.87, 0.90, 1.0)
    bg.inputs['Strength'].default_value = 0.35
    scn.camera = cam
    scn.render.fps = 30
    scn.frame_start, scn.frame_end = 1, n_frames
    scn.frame_set(1)
    for o in (pivot, look):                                           # camera helpers: not drawn
        o.hide_set(True)
    for o in bpy.data.objects:
        o.select_set(False)
    for screen in bpy.data.screens:                                   # open looking through the camera, with materials
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                sp = area.spaces.active
                sp.shading.type = 'MATERIAL'
                sp.overlay.show_floor = False
                sp.overlay.show_axis_x = sp.overlay.show_axis_y = False
                sp.region_3d.view_perspective = 'CAMERA'


def main():
    a = parse_args()
    for o in list(bpy.data.objects):                                  # start from an empty scene
        bpy.data.objects.remove(o, do_unlink=True)
    body = SMPLX(a.smplx) if a.smplx else None
    skin = material('body', (0.75, 0.75, 0.75, 1.0), 0.5)
    blue = material('route', (0.35, 0.55, 0.85, 1.0))
    first, n_frames = None, 1
    for k, path in enumerate(a.npz):
        z = np.load(path, allow_pickle=True)
        quats = np.asarray(z['quats'], np.float64)
        root = np.asarray(z['root'], np.float64)
        name = Path(path).name.split('.')[0]
        if body is not None:
            verts, joints = body.rest(z['betas'])
        else:
            joints = rest_from_offsets(z['skel_offset'])
        arm = build_armature(name, joints)
        offset_x = k * a.spacing
        arm.location.x += offset_x
        if body is not None:
            build_mesh(name + '_body', verts, body.faces, body.weights, arm, skin)
        else:
            build_mesh(name + '_body', *stick_figure(joints), arm, skin)
        arm.hide_set(True)
        animate(arm, quats, root, joints)
        if not a.no_route and 'cmd_xz_m' in z.files:
            build_route(name + '_route', np.asarray(z['cmd_xz_m'])[:len(quats)], offset_x, blue)
        first = first or arm
        n_frames = max(n_frames, len(quats))
        print(f'{name}: {len(quats)} frames' + (' (SMPL-X mesh)' if body is not None else ' (skeleton)'))
    build_stage(first, n_frames)
    if a.out:
        Path(a.out).resolve().parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.out).resolve()))
        print(f'saved {a.out}')


if __name__ == '__main__':
    main()
