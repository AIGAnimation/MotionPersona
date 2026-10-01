"""Offline generation with the released model: persona x style x body along a trajectory -> BVH.

    python generate.py --persona p02 --style happy                        # p02 on its own body, 20 s of the eval route
    python generate.py --persona p02,p10 --style neutral,drunk --body grid_h08_g04 --seconds 30
    python generate.py --persona p05 --style angry --body 72 --route video_straight --seeds 0,1,2

Every case starts from the same canonical walking history placed on the chosen body and follows a world-fixed,
time-indexed route (speed, heading and facing over time) scaled by the body's leg length, generated block by block
as in the paper: two Euler steps per block, `--hop` frames committed per block after a `--blend`-frame cross-fade
with the previous block, the committed frames becoming the next block's history. The output is the raw model output;
`--foot-lock` additionally writes a foot-locked copy (toe locking + two-bone leg IK, utils/foot_lock.py).

Outputs per case in --out:  <persona>__<body>__<style>__s<seed>.bvh  (30 fps, centimetres)  and  .npz  (quaternions,
root, foot contacts, skeleton offsets, betas, the commanded route); with --foot-lock also
<stem>.footlock.bvh.
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE
sys.path.insert(0, str(SRC))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data.loco_dataset import STYLE_VOCAB, SUBJECT_VOCAB  # noqa: E402
from evaluation.archive import write_bvh  # noqa: E402
from evaluation.cases import Case, load_bodies, own_body  # noqa: E402
from evaluation.engine import Engine  # noqa: E402
from evaluation.heads import load_bundle, make_head  # noqa: E402
from evaluation.route import Route  # noqa: E402
from utils.weights import DEFAULT_DIR, ensure_weights  # noqa: E402

ROUTES = {'eval': SRC / 'data/eval/eval_traj.npz',                 # the one-minute evaluation route of the paper
          'video_straight': SRC / 'data/eval/video_straight.npz',   # straight walk, speed changes only
          'switch': SRC / 'data/eval/switch_traj.npz'}              # walk -> run -> walk
STYLES = [s for s in STYLE_VOCAB if s != 't_pose']


def resolve_body(spec, persona, bodies):
    """'own' -> the persona's captured body; an integer -> bodies.csv id; otherwise a body name (p01, grid_h03_g00, ...)."""
    if spec == 'own':
        return own_body(persona, bodies)
    if spec.isdigit():
        if int(spec) not in bodies:
            raise SystemExit(f'unknown body id {spec} (0..{max(bodies)})')
        return int(spec)
    for b, r in bodies.items():
        if r['name'] == spec:
            return b
    raise SystemExit(f'unknown body {spec!r}; see data/bodies/bodies.csv')


def lock_feet(T_pose, offsets, quats, root, mode, fps):
    """Foot-locked copy of one clip (utils/foot_lock.py works in centimetres; the rollout is in metres)."""
    from utils.foot_lock import foot_lock
    res = foot_lock(quats, np.asarray(root) * 100.0, np.asarray(offsets) * 100.0, T_pose.parents, T_pose.names,
                    1.0 / fps, cfg={'mode': mode})
    return res.rotations.astype(np.float32), (res.root_pos / 100.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default=str(DEFAULT_DIR / 'prior.ckpt'), help='stage-2 checkpoint (default: the released model, downloaded on first use)')
    ap.add_argument('--persona', default='p02', help=f'comma list of performer IDs ({SUBJECT_VOCAB[0]}..{SUBJECT_VOCAB[-1]}) or "all"')
    ap.add_argument('--style', default='neutral', help=f'comma list of {", ".join(STYLES)} or "all"')
    ap.add_argument('--body', default='own', help='comma list: "own" (the persona\'s own body), a body id or name from bodies.csv')
    ap.add_argument('--route', default='eval', help='eval | video_straight | switch | path to a route .npz')
    ap.add_argument('--seconds', type=float, default=20.0, help='length of the generated motion (<= 60 s on the eval route)')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--hop', type=int, default=12, help='frames committed per block (paper: 12)')
    ap.add_argument('--blend', type=int, default=2, help='cross-fade frames at each commit (paper: 2)')
    ap.add_argument('--steps', type=int, default=2, help='Euler steps per block (paper: 2)')
    ap.add_argument('--guidance', type=float, default=1.0, help='history guidance weight (1 = off)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--no-ema', action='store_true', help='use the raw weights instead of the EMA weights')
    ap.add_argument('--foot-lock', choices=('off', 'runtime', 'offline'), default='off',
                    help='also write <stem>.footlock.bvh: runtime = causal lock as in the realtime demo, offline = whole-clip solve')
    ap.add_argument('--out', default=str(HERE / 'outputs'))
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    bodies = load_bodies(SRC / 'data/bodies/bodies.csv')
    personas = list(SUBJECT_VOCAB) if a.persona == 'all' else a.persona.split(',')
    styles = STYLES if a.style == 'all' else a.style.split(',')
    for p in personas:
        if p not in SUBJECT_VOCAB:
            raise SystemExit(f'unknown persona {p!r}; choose from {SUBJECT_VOCAB}')
    for s in styles:
        if s not in STYLES:
            raise SystemExit(f'unknown style {s!r}; choose from {STYLES}')
    seeds = [int(s) for s in a.seeds.split(',')]
    cases = [Case(p, resolve_body(b, p, bodies), s, seed)
             for p in personas for b in a.body.split(',') for s in styles for seed in seeds]

    route = Route.load(str(ROUTES.get(a.route, a.route)))
    n_frames = int(round(a.seconds * route.fps)) // a.hop * a.hop
    if Path(a.ckpt) == DEFAULT_DIR / 'prior.ckpt':
        ensure_weights()
    bundle = load_bundle(a.ckpt, device=a.device, ema=not a.no_ema, steps=a.steps, hist_guidance=a.guidance,
                         tpose_dir=SRC / 'data/skeleton')
    head = make_head(bundle)
    need = 1 + n_frames + (bundle.F - a.hop)
    if need > route.n:
        raise SystemExit(f'--seconds too long for this route (max {(route.n - 1 - bundle.F + a.hop) / route.fps:.1f} s)')
    eng = Engine(bundle, head, route, SRC / 'data/eval/canonical', caps_csv=SRC / 'data/eval/speed_caps.csv',
                 bodies=bodies, hop=a.hop, n_frames=n_frames, batch_size=min(a.batch_size, len(cases)), blend=a.blend)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f'{len(cases)} case(s), {n_frames} frames each ({n_frames / route.fps:.1f} s), device {a.device}')
    t0 = time.time()
    for res in eng.run(cases):
        c = res.case
        stem = f'{c.persona}__{bodies[c.body]["name"]}__{c.style}__s{c.seed}'
        write_bvh(out / f'{stem}.bvh', bundle.T_pose, res.skel_offset, res.quats, res.root)
        if a.foot_lock != 'off':
            q, r = lock_feet(bundle.T_pose, res.skel_offset, res.quats, res.root, a.foot_lock, route.fps)
            write_bvh(out / f'{stem}.footlock.bvh', bundle.T_pose, res.skel_offset, q, r)
        np.savez(out / f'{stem}.npz', quats=res.quats, root=res.root, contact=res.contact, skel_offset=res.skel_offset,
                 betas=res.betas, cmd_xz_m=res.cmd_xz_m, cmd_speed_mps=res.cmd_speed_mps, cmd_heading=res.cmd_heading,
                 cmd_facing=res.cmd_facing, fps=route.fps)
        print(f'  {stem}  ({bodies[c.body]["height"][:4]} m tall, block {np.median(res.block_ms):.1f} ms)')
    print(f'done in {time.time() - t0:.1f} s -> {out}')


if __name__ == '__main__':
    main()
