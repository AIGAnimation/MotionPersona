"""One-minute rollouts along the evaluation route, archived for evaluation/run_metrics.py.

    python scripts/eval_rollout.py --test-set own --out save/eval/released/own            # 44 personas x 9 styles x 5 seeds
    python scripts/eval_rollout.py --test-set seen --out save/eval/released/seen          # also: withheld, heldout
    python scripts/eval_rollout.py --test-set own1:p02 --seeds 0 --out save/eval/dev/own1_p02   # one performer, all styles
    python scripts/eval_rollout.py --cases cases.csv --set-name mine --out ...             # persona,body,style,seed rows

The paper's protocol is the default: the evaluation route (1,800 frames), the canonical history on the test body, two
Euler steps per block, 12 frames committed per block after a 2-frame linear cross-fade, EMA weights, no foot locking.
`--ckpt` takes the released prior (default, downloaded on first use), a run directory (best.ckpt) or a checkpoint.

Determinism: fixed batch shape, deterministic algorithms, TF32 off; noise per (seed, case, block) on the CPU, so a
case's output does not depend on which cases share its batch (same device and batch size => identical output).
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.archive import Archive  # noqa: E402
from evaluation.cases import SEEDS, build_cases, cases_from_csv, load_bodies  # noqa: E402
from evaluation.engine import Engine, token_flags  # noqa: E402
from evaluation.heads import load_bundle, make_head  # noqa: E402
from evaluation.route import Route, file_sha256  # noqa: E402
from utils.weights import DEFAULT_DIR, ensure_weights  # noqa: E402

ROUTES = {'eval': REPO / 'data/eval/eval_traj.npz', 'switch': REPO / 'data/eval/switch_traj.npz',
          'video_straight': REPO / 'data/eval/video_straight.npz'}


def text_lookup_for(bundle):
    """(persona, style) -> CLIP row for checkpoints conditioned on a persona prompt (realtime/assets/text_feats.npz)."""
    if token_flags(bundle.module)[0] != 'text':
        return None
    from realtime.model_loop import TextFeats
    tf = TextFeats()
    return lambda p, s: tf.Get(p, s, 0)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default=str(DEFAULT_DIR / 'prior.ckpt'), help='stage-2 checkpoint or run directory (default: the released model)')
    ap.add_argument('--test-set', default=None, help='own | seen | withheld | heldout | own_neutral | own1:<persona> | sweep_shape | sweep_style | sweep_persona')
    ap.add_argument('--cases', default=None, help='CSV with persona,body,style,seed columns (instead of --test-set)')
    ap.add_argument('--set-name', default=None, help="test_set recorded in the archive for --cases runs (default 'cases')")
    ap.add_argument('--split', default=str(REPO / 'data/eval/split_v2'))
    ap.add_argument('--route', default='eval', help='eval | switch | video_straight | path to a route .npz')
    ap.add_argument('--canonical', default=str(REPO / 'data/eval/canonical'))
    ap.add_argument('--caps', default=str(REPO / 'data/eval/speed_caps.csv'))
    ap.add_argument('--bodies', default=str(REPO / 'data/bodies/bodies.csv'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default=','.join(map(str, SEEDS))); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--hop', type=int, default=12, help='frames committed per block (paper: 12)')
    ap.add_argument('--blend', type=int, default=2, help='cross-fade frames at each commit (paper: 2)')
    ap.add_argument('--blend-shape', default='linear', choices=['linear', 'smoothstep'])
    ap.add_argument('--frames', type=int, default=None, help="default: the route's evaluated length (1800 on the evaluation route)")
    ap.add_argument('--steps', type=int, default=None, help='Euler steps per block (default 2, the paper)')
    ap.add_argument('--sampler', default=None, help='euler (default) | heun')
    ap.add_argument('--guidance', type=float, default=1.0, help='history guidance weight (1 = off)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--no-ema', action='store_true', help='use the raw weights instead of the EMA weights')
    ap.add_argument('--resume', action='store_true', help='skip cases already in --out')
    a = ap.parse_args()
    if not a.test_set and not a.cases:
        raise SystemExit('pass --test-set or --cases')

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(a.threads))
    if Path(a.ckpt) == DEFAULT_DIR / 'prior.ckpt':
        ensure_weights()
    bundle = load_bundle(a.ckpt, device=a.device, ema=not a.no_ema, steps=a.steps, sampler=a.sampler, hist_guidance=a.guidance)
    bodies = load_bodies(a.bodies)
    route_path = str(ROUTES.get(a.route, a.route))
    route = Route.load(route_path)
    if a.frames is None:
        a.frames = int(route.n_eval)
    seeds = tuple(int(s) for s in a.seeds.split(','))
    cases = cases_from_csv(a.cases) if a.cases else build_cases(a.test_set, bodies=bodies, split_dir=a.split, seeds=seeds)
    if a.limit:
        cases = cases[:a.limit]
    head = make_head(bundle)
    eng = Engine(bundle, head, route, a.canonical, caps_csv=a.caps, bodies=bodies, hop=a.hop, n_frames=a.frames,
                 batch_size=a.batch_size, text_lookup=text_lookup_for(bundle), blend=a.blend, blend_shape=a.blend_shape)
    canon_index = Path(a.canonical) / 'index.json'
    ck = Path(a.ckpt)
    run_name = ck.name if ck.is_dir() else (ck.stem if ck.parent == DEFAULT_DIR else ck.parent.name)
    settings = dict(steps=getattr(head, 'steps', None), sampler=getattr(head, 'sampler', None), hop=a.hop, blend=a.blend,
                    blend_shape=a.blend_shape, batch_size=a.batch_size, device=a.device, torch=torch.__version__,
                    cuda=torch.version.cuda, threads=a.threads, deterministic_algorithms=True, tf32=False,
                    hist_guidance=a.guidance, frames=a.frames)
    if a.route != 'eval':
        settings.update(route=route_path, route_version=route.version)        # recorded only off the default route
    arch = Archive(a.out, bundle, run_name=run_name, test_set=a.test_set or a.set_name or 'cases', route_sha=route.sha256,
                   canonical_index_sha=file_sha256(canon_index) if canon_index.exists() else None, caps_sha=eng.caps_sha,
                   persona_rows_sha=file_sha256(REPO / 'realtime/assets/persona_rows.json'), settings=settings)
    if a.resume:
        before = len(cases); cases = [c for c in cases if not arch.done(c.key)]
        print(f'resume: {before - len(cases)} cases already written, {len(cases)} to go')
    print(f'{run_name}: {bundle.kind} head, {len(cases)} cases, batch {a.batch_size}, {a.frames} frames, device {a.device}')
    t0 = time.time(); n = 0
    for res in eng.run(cases):
        arch.write(res); n += 1
        if n % a.batch_size == 0 or n == len(cases):
            arch.flush_manifest()
            print(f'  {n}/{len(cases)} written  ({(time.time() - t0) / 60:.1f} min)')
    p = arch.flush_manifest()
    print(f'done: {n} cases -> {a.out} ({p.name}); {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
