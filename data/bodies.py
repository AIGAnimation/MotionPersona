"""Target bodies for retargeting: fast betas -> skeleton, and the height x girth body grid.

SMPL-X joint locations are LINEAR in the shape coefficients:  J(beta) = J0 + A @ beta  with
A = J_regressor @ shapedirs (55,3,10), J0 = J_regressor @ v_template.  The 23-joint repo skeleton (flat_v1
`skel_offset`) is therefore a closed-form function of betas -- no mesh, no smplx at runtime.  The mesh is only
needed once, to build `joint_regressor.npz` and to calibrate the grid (height / volume of a body).

    reg = load_regressor()                              # data/bodies/joint_regressor.npz (built by build_regressor)
    offs = betas_to_offsets(betas, reg)                 # (N,23,3) metres; root row per the ground convention (betas_to_offsets):
    #   y=0 is the rigid foot-model corner plane; skel_offset[0] = pelvis above the MESH ground (sole basis, default),
    #   which coincides with the corner plane at rest; a rendered mesh stands ~0.5-1.0 cm above y=0 in motion.

Grid coordinates: `u_h` = unit beta direction of the mesh-height gradient (~ beta0),
`u_g` = unit direction of the mesh-volume gradient orthogonal to u_h (~ 0.81 b1 + 0.46 b2 + 0.30 b3): along u_g
the height stays fixed while the volume (hence BMI) changes.  Real bodies sit at t_h in [-6.3, 1.9]
(105-191 cm) and t_g in [-1.7, 1.4] (no obese subjects); the designed grid spans BMI 14..40.
"""
import csv
from pathlib import Path

import numpy as np

REPO_NAMES = ['pelvis', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'right_hip', 'right_knee', 'right_ankle', 'right_foot',
              'spine1', 'spine2', 'spine3', 'neck', 'head', 'jaw', 'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist',
              'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist']
REPO_PARENTS = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 11, 15, 16, 17, 11, 19, 20, 21])
BODIES_DIR = Path(__file__).resolve().parent / 'bodies'   # repo asset: resolve from this file, not the cwd
                                                          # (the realtime demo runs from realtime/)
REGRESSOR = BODIES_DIR / 'joint_regressor.npz'
DENSITY = 1000.0


# ----------------------------------------------------------------------------- regressor (needs smplx once)
def build_regressor(model_dir, out=REGRESSOR):
    """Extract the linear joint map and the grid directions from SMPLX_NEUTRAL; saves and returns the dict."""
    import smplx
    from smplx.joint_names import JOINT_NAMES
    m = smplx.create(model_dir, model_type='smplx', gender='neutral', num_betas=10, use_pca=False, flat_hand_mean=True, batch_size=1)
    J_reg = m.J_regressor.detach().numpy()[:55]
    S = m.shapedirs.detach().numpy()                    # (V,3,10)
    V0 = m.v_template.detach().numpy()                  # (V,3)
    F = m.faces.astype(np.int64)
    A = np.einsum('jv,vck->jck', J_reg, S)
    J0 = J_reg @ V0
    repo_idx = np.array([JOINT_NAMES.index(n) for n in REPO_NAMES])
    # grid directions from the (linear) mesh: height gradient and volume gradient at the mean body
    def height(b):
        V = V0 + S @ b
        return V[:, 1].max() - V[:, 1].min()
    def volume(b):
        V = V0 + S @ b
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        return abs((v0 * np.cross(v1, v2)).sum() / 6.0)
    eye = np.eye(10)
    g_h = np.array([(height(0.1 * e) - height(-0.1 * e)) / 0.2 for e in eye])
    g_v = np.array([(volume(0.1 * e) - volume(-0.1 * e)) / 0.2 for e in eye])
    u_h = g_h / np.linalg.norm(g_h)
    g_perp = g_v - (g_v @ u_h) * u_h
    u_g = g_perp / np.linalg.norm(g_perp)
    reg = dict(A=A.astype(np.float64), J0=J0.astype(np.float64), repo_idx=repo_idx, parents=REPO_PARENTS, names=np.array(REPO_NAMES),
               u_h=u_h, u_g=u_g, height0=height(np.zeros(10)), volume0=volume(np.zeros(10)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **reg)
    return reg


def load_regressor(path=REGRESSOR):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- betas -> skeleton (no mesh)
def betas_to_joints(betas, reg):
    """(N,10) -> (N,55,3) SMPL-X rest joints (SMPL-X frame, pelvis not grounded)."""
    b = np.atleast_2d(np.asarray(betas, dtype=np.float64))
    return reg['J0'][None] + np.einsum('jck,nk->njc', reg['A'], b)


_SOLE_BASIS = None


def _sole_basis():
    global _SOLE_BASIS
    if _SOLE_BASIS is None:
        z = np.load(BODIES_DIR / 'sole_basis.npz')
        _SOLE_BASIS = {'c0': float(z['c0']), 'c': np.asarray(z['c'], np.float64)}
    return _SOLE_BASIS


def betas_to_offsets(betas, reg, root_basis='sole'):
    """(N,10) -> (N,23,3) flat_v1 skel_offset: bone offsets to the parent.

    Ground convention: y = 0 is the rigid foot-model corner plane; skel_offset[0] = pelvis above the MESH ground
    (sole basis), constant per body; a rendered SMPL-X mesh stands with its sole approximately +0.5-1.0 cm above
    y=0 (body-independent).  root_basis='sole' (default) implements that via the linear sole-drop map fitted on the
    128 bank bodies (data/bodies/sole_basis.npz, rms 0.02 cm; mesh ground == corner plane at rest). 'joint' = the
    alternative root row (pelvis above the lowest foot JOINT, ~0.9-1.9 cm lower); the training data
    (data.skel_root_basis) and the runtime must use the same basis."""
    J = betas_to_joints(betas, reg)[:, reg['repo_idx']]          # (N,23,3)
    par = reg['parents']
    off = J - J[:, np.clip(par, 0, None)]
    lf, rf = REPO_NAMES.index('left_foot'), REPO_NAMES.index('right_foot')
    ground = np.minimum(J[:, lf, 1], J[:, rf, 1])
    if root_basis == 'sole':
        sb = _sole_basis()
        bb = np.atleast_2d(np.asarray(betas, dtype=np.float64))
        ground = ground - (sb['c0'] + bb @ sb['c'])
    elif root_basis != 'joint':
        raise ValueError(f"root_basis must be 'sole' or 'joint', got {root_basis!r}")
    off[:, 0] = J[:, 0] - np.stack([np.zeros_like(ground), ground, np.zeros_like(ground)], -1)
    return off.astype(np.float32)


def skeleton_stats(offsets):
    """Per body: leg length, hip width, hip height (pelvis to toe, T-pose), skeleton height (head above lowest foot)."""
    J = REPO_NAMES.index
    o = np.asarray(offsets, dtype=np.float64)
    leg = np.linalg.norm(o[:, J('left_knee')], axis=-1) + np.linalg.norm(o[:, J('left_ankle')], axis=-1)
    hip_w = np.linalg.norm(o[:, J('left_hip')] - o[:, J('right_hip')], axis=-1)
    hip_h = np.abs(o[:, J('left_hip'), 1] + o[:, J('left_knee'), 1] + o[:, J('left_ankle'), 1] + o[:, J('left_foot'), 1])
    # T-pose positions by accumulating offsets (root at its offset)
    pos = np.zeros_like(o)
    for j, p in enumerate(REPO_PARENTS):
        pos[:, j] = o[:, j] if p < 0 else pos[:, p] + o[:, j]
    height = pos[:, J('head'), 1] - np.minimum(pos[:, J('left_foot'), 1], pos[:, J('right_foot'), 1])
    return dict(leg=leg, hip_w=hip_w, hip_h=hip_h, skel_height=height)


# ----------------------------------------------------------------------------- grid (needs smplx once)
class MeshProbe:
    """Height / volume of the SMPL-X mesh as a function of betas (linear mesh, exact)."""
    def __init__(self, model_dir):
        import smplx
        m = smplx.create(model_dir, model_type='smplx', gender='neutral', num_betas=10, use_pca=False, flat_hand_mean=True, batch_size=1)
        self.S = m.shapedirs.detach().numpy(); self.V0 = m.v_template.detach().numpy(); self.F = m.faces.astype(np.int64)

    def mesh(self, b):
        return self.V0 + self.S @ np.asarray(b, dtype=np.float64)

    def height(self, b):
        V = self.mesh(b); return float(V[:, 1].max() - V[:, 1].min())

    def volume(self, b):
        V = self.mesh(b); v0, v1, v2 = V[self.F[:, 0]], V[self.F[:, 1]], V[self.F[:, 2]]
        return float(abs((v0 * np.cross(v1, v2)).sum() / 6.0))

    def bmi(self, b):
        h = self.height(b); return DENSITY * self.volume(b) / (h * h)


def make_grid(reg, probe, n_h=10, n_g=8, heights=(1.05, 1.95), bmis=(14.0, 40.0)):
    """Grid bodies beta = t_h*u_h + t_g*u_g: t_h solved so the mesh height hits the targets (height is linear in t_h),
    t_g by bisection so the mesh-BMI (1000*volume/height^2) hits the targets at that height."""
    u_h, u_g = reg['u_h'], reg['u_g']
    h_targets = np.linspace(heights[0], heights[1], n_h)
    b_targets = np.linspace(bmis[0], bmis[1], n_g)
    # height(t_h) is linear: two probes give the slope
    h0, h1 = probe.height(0 * u_h), probe.height(1.0 * u_h)
    rows = []
    for i, ht in enumerate(h_targets):
        t_h = (ht - h0) / (h1 - h0)
        for j, bt in enumerate(b_targets):
            lo, hi = -6.0, 12.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if probe.bmi(t_h * u_h + mid * u_g) < bt:
                    lo = mid
                else:
                    hi = mid
            t_g = 0.5 * (lo + hi)
            beta = t_h * u_h + t_g * u_g
            rows.append(dict(name=f'grid_h{i:02d}_g{j:02d}', kind='grid', i_height=i, j_girth=j, t_h=t_h, t_g=t_g, betas=beta,
                             height=probe.height(beta), volume=probe.volume(beta), bmi_proxy=probe.bmi(beta)))
    return rows


def real_bodies(shapes_dir, probe, reg, meta_csv='data/_raw/meta.csv'):
    meta = {}
    if Path(meta_csv).exists():
        meta = {r['Name'].strip().lower(): r for r in csv.DictReader(open(meta_csv, encoding='utf-8-sig'))}
    rows = []
    for f in sorted(Path(shapes_dir).glob('*.npz')):
        if '_' in f.stem:
            continue
        beta = np.load(f)['betas'][:10].astype(np.float64)
        m = meta.get(f.stem, {})
        rows.append(dict(name=f.stem, kind='real', i_height=-1, j_girth=-1, t_h=float(beta @ reg['u_h']), t_g=float(beta @ reg['u_g']), betas=beta,
                         height=probe.height(beta), volume=probe.volume(beta), bmi_proxy=probe.bmi(beta),
                         meta_height_cm=float(m['Height']) if m else np.nan, meta_weight_kg=float(m['Weight']) if m else np.nan,
                         meta_age=float(m['Age']) if m else np.nan, meta_gender=m.get('Gender', '').strip().lower() if m else ''))
    return rows


def write_bodies(rows, reg, out_dir=BODIES_DIR):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    betas = np.array([r['betas'] for r in rows])
    offs = betas_to_offsets(betas, reg)
    st = skeleton_stats(offs)
    for k, r in enumerate(rows):
        r['id'] = k
        for key in ('leg', 'hip_w', 'hip_h', 'skel_height'):
            r[key] = float(st[key][k])
        r['mass'] = DENSITY * r['volume']
    fields = ['id', 'name', 'kind', 'i_height', 'j_girth', 't_h', 't_g', 'height', 'skel_height', 'volume', 'mass', 'bmi_proxy', 'leg', 'hip_w', 'hip_h',
              'meta_height_cm', 'meta_weight_kg', 'meta_age', 'meta_gender'] + [f'b{i}' for i in range(10)]
    with open(out_dir / 'bodies.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            row = {k: r.get(k, '') for k in fields[:19]}
            row.update({f'b{i}': f'{r["betas"][i]:.5f}' for i in range(10)})
            for k in ('t_h', 't_g', 'height', 'skel_height', 'volume', 'mass', 'bmi_proxy', 'leg', 'hip_w', 'hip_h'):
                row[k] = f'{r[k]:.5f}'
            w.writerow(row)
    np.savez(out_dir / 'bodies.npz', id=np.array([r['id'] for r in rows]), name=np.array([r['name'] for r in rows]), kind=np.array([r['kind'] for r in rows]),
             betas=betas.astype(np.float32), offsets=offs, t_h=np.array([r['t_h'] for r in rows]), t_g=np.array([r['t_g'] for r in rows]),
             height=np.array([r['height'] for r in rows]), mass=np.array([r['mass'] for r in rows]), bmi_proxy=np.array([r['bmi_proxy'] for r in rows]),
             leg=st['leg'], hip_w=st['hip_w'], hip_h=st['hip_h'], skel_height=st['skel_height'])
    return out_dir / 'bodies.csv'
