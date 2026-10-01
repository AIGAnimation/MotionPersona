"""Batched one-minute rollouts.

A test case is (persona, body, style, seed). Every case follows the same world-fixed, time-indexed route
(evaluation/route.py) scaled by its body's leg, starts from the same canonical history on that body
(evaluation/canonical.py), and is generated block by block: the plan of block b is the route's frames hop*b+1 ..
hop*b+F relative to the ACTUAL root of the last committed frame; the head samples K+F frames; the first `hop` (3)
future frames are committed and become the newest history. No cross-fade, no foot locking, no post-processing.

Per-block conditioning is exactly realtime/model_loop.py::PersonaModel.generate's batch, built for B cases at once.

Noise is a pure function of (seed, case key, block, draw): each case draws its own CPU `torch.randn` from a generator
seeded by `noise_seed`, and the draws are stacked into the batch -- so a case's noise never depends on which other
cases share its batch. GPU kernels do depend on the batch shape, so batches are padded to a fixed `batch_size`
(padding rows are repeats of the last case and are discarded): same (run, case, seed, batch_size, device) => identical
output; B=1 vs B=batch_size agree to float tolerance (tests).
"""
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from data.bodies import REGRESSOR, betas_to_offsets, load_regressor
from data.loco_dataset import STYLE_INDEX
from evaluation.cases import Case
from evaluation.route import Route, block_plan, file_sha256
from realtime.model_loop import PersonaRows, hip_forward
from utils.nn_transforms import repr6d2quat

TEXT_DIM = 512


def noise_seed(seed, key, block, draw):
    """Deterministic 63-bit seed for one (case seed, case key, block, draw index)."""
    ss = np.random.SeedSequence([int(seed), int(block), int(draw), zlib.crc32(str(key).encode())])
    return int(ss.generate_state(1, np.uint64)[0] >> np.uint64(1))


def draw_noise(shape, seeds):
    """Stack one CPU randn per seed -> (len(seeds), *shape); each row depends on its seed only."""
    return torch.stack([torch.randn(tuple(shape), generator=torch.Generator().manual_seed(int(s))) for s in seeds])


@dataclass
class CaseResult:
    case: Case
    quats: np.ndarray        # (n, J, 4) wxyz, world orientation for the root
    root: np.ndarray         # (n, 3) metres, world; y above the sole plane
    contact: np.ndarray      # (n, 2) model output
    skel_offset: np.ndarray  # (J, 3) metres
    betas: np.ndarray        # (10,)
    leg: float
    speed_scale: float
    v_cap_leg: float | None
    cmd_xz_m: np.ndarray     # (n, 2) the scaled route, rows 1..n
    cmd_speed_mps: np.ndarray
    cmd_heading: np.ndarray
    cmd_facing: np.ndarray
    canonical: dict
    block_ms: np.ndarray     # (n_blocks,) wall ms per block of THE BATCH the case ran in
    batch_size: int
    batch_index: int
    extra: dict = field(default_factory=dict)


def token_flags(module):
    """(persona_cond, use_style) of a loaded module. The latent prior keeps them on the LightningModule; the raw DDPM /
    FM modules keep them on the inner denoiser (`module.model`, network/models.py), which is where cond_token_kwargs
    reads them."""
    for obj in (module, getattr(module, 'model', None)):
        if obj is None:
            continue
        if hasattr(obj, 'persona_cond') or hasattr(obj, 'use_style'):
            return str(getattr(obj, 'persona_cond', 'text')), bool(getattr(obj, 'use_style', False))
    return 'text', False


class Engine:
    def __init__(self, bundle, head, route: Route, canonical_dir, *, caps_csv=None, bodies=None, hop=3, n_frames=1800,
                 batch_size=256, persona_rows=None, text_lookup=None, route_leg_override=None, switch=None, blend=0,
                 blend_shape='linear', log=print):
        self.b, self.head, self.route = bundle, head, route
        self.canonical_dir = Path(canonical_dir)
        self.hop, self.n_frames, self.batch_size = int(hop), int(n_frames), int(batch_size)
        self.n_blocks = self.n_frames // self.hop
        assert self.n_blocks * self.hop == self.n_frames
        assert 1 + self.n_blocks * self.hop + (bundle.F - self.hop) <= route.n, 'route too short for the block plan'
        self.bodies = bodies or {}
        self.caps = {}
        if caps_csv:
            import csv
            self.caps = {int(r['body']): float(r['v_p95_leg']) for r in csv.DictReader(open(caps_csv))}
        self.caps_sha = file_sha256(caps_csv) if caps_csv else None
        self.rows = persona_rows or PersonaRows()
        self.text_lookup = text_lookup          # callable (persona, style) -> (512,) for persona_cond=text checkpoints
        self.reg = load_regressor(REGRESSOR)
        self.route_leg_override = route_leg_override   # metres: scale the route by this leg on every body (3.8)
        self.switch = switch                           # (style, block): from that block on every case carries this style (tab:guidance)
        self.blend = int(blend)                       # frames crossfaded from the previous block's continuation into the new one
        assert self.blend < self.hop or self.blend == 0, 'blend must be < hop (otherwise no committed frame is ever fully the new block)'
        self.blend_shape = str(blend_shape)           # linear (the demo, default) | smoothstep (C1 across the seam)
        # Not implemented: committing window frames [lead, lead+hop) instead of [0, hop). The first frames of a decoded
        # window are slightly rougher than its middle, but the change needs a pivot `lead` frames back, which has no
        # route rows before the origin for the first block.
        self.log = log
        m = bundle.module
        self.rp_mean = m.root_pos_mean.detach().clone().to(bundle.device)
        self.rp_std = m.root_pos_std.detach().clone().to(bundle.device)
        self._canon = {}

    # ------------------------------------------------------------------ per-case inputs
    def leg(self, body):
        return float(self.bodies[body]['leg'])

    def canonical(self, body):
        if body not in self._canon:
            p = self.canonical_dir / f'{body:03d}.npz'
            if not p.exists():
                raise SystemExit(f'canonical history for body {body} missing: {p} (shipped in data/eval/canonical/)')
            z = np.load(p)
            self._canon[body] = dict(rotations=z['rotations'], root_pos=z['root_pos'], foot_contact=z['foot_contact'],
                                     take=str(z['take']), start=int(z['start']), end=int(z['end']), speed_leg=float(z['speed_leg']),
                                     sha256=file_sha256(p), path=str(p))
        return self._canon[body]

    def conditions(self, case):
        b = self.bodies[case.body]
        betas = np.array([float(b[f'b{i}']) for i in range(10)], np.float32)
        skel = betas_to_offsets(betas[None].astype(np.float64), self.reg, root_basis=self.b.root_basis)[0].astype(np.float32)
        out = dict(betas=betas, skel=skel)
        pc, use_style = token_flags(self.b.module)
        if pc in ('id', 'id_attr', 'attr'):
            sid, attr = self.rows.Get(case.persona)
            out['subject_idx'], out['attr_idx'] = np.int64(sid), np.asarray(attr, np.int64)
        if use_style:
            out['style_idx'] = np.int64(STYLE_INDEX[case.style])
        if pc == 'text':
            if self.text_lookup is None:
                raise SystemExit('this checkpoint takes a CLIP text token: pass text_lookup=(persona, style) -> (512,)')
            out['text_feat'] = np.asarray(self.text_lookup(case.persona, case.style), np.float32)
        else:
            out['text_feat'] = np.zeros(TEXT_DIM, np.float32)         # featurize_batch indexes it; the model ignores it
        return out

    # ------------------------------------------------------------------ the loop
    def run(self, cases):
        """Yield a CaseResult per case, in order, batch by batch."""
        cases = list(cases)
        for bi in range(0, len(cases), self.batch_size):
            chunk = cases[bi:bi + self.batch_size]
            for res in self._run_batch(chunk, bi // self.batch_size):
                yield res

    def _run_batch(self, chunk, batch_index):
        dev = self.b.device
        K, F, J, hop = self.b.K, self.b.F, self.b.J, self.hop
        B0 = len(chunk)
        pad = [chunk[-1]] * (self.batch_size - B0)           # fixed batch shape; padded rows are discarded
        cs = chunk + pad
        B = len(cs)
        # per-case static inputs
        conds = [self.conditions(c) for c in cs]
        legs = [self.leg(c.body) for c in cs]
        scaled = []
        for c, leg in zip(cs, legs):
            leg_r = leg if self.route_leg_override is None else float(self.route_leg_override)
            scaled.append(self.route.scaled(leg_r, self.caps.get(c.body)))
        xz_m = torch.tensor(np.stack([s.xz_m for s in scaled]), dtype=torch.float32, device=dev)       # (B, N, 2)
        quat = torch.tensor(np.stack([s.quat for s in scaled]), dtype=torch.float32, device=dev)       # (B, N, 4)
        canon = [self.canonical(c.body) for c in cs]
        hq = torch.tensor(np.stack([z['rotations'][-K:] for z in canon]), dtype=torch.float32, device=dev)   # (B, K, J, 4)
        hr = torch.tensor(np.stack([z['root_pos'][-K:] for z in canon]), dtype=torch.float32, device=dev)    # (B, K, 3)
        hc = torch.tensor(np.stack([z['foot_contact'][-K:] for z in canon]), dtype=torch.float32, device=dev)  # (B, K, 2)
        static = {
            'aug_trig': torch.tensor([[1.0, 0.0, 1.0, 0.0]] * B, device=dev),
            'shape_feat': torch.tensor(np.stack([c['betas'] for c in conds]), device=dev),
            'skel_offset': torch.tensor(np.stack([c['skel'] for c in conds]), device=dev),
            'text_feat': torch.tensor(np.stack([c['text_feat'] for c in conds]), device=dev),
            'motion_idx': torch.zeros(B, dtype=torch.int64, device=dev), 'clip_start': torch.zeros(B, dtype=torch.int64, device=dev),
        }
        for k in ('subject_idx', 'attr_idx', 'style_idx'):
            if k in conds[0]:
                static[k] = torch.tensor(np.stack([c[k] for c in conds]), device=dev)
        out_q = torch.zeros(B, self.n_frames, J, 4, device=dev)
        out_r = torch.zeros(B, self.n_frames, 3, device=dev)
        out_c = torch.zeros(B, self.n_frames, 2, device=dev)
        block_ms = np.zeros(self.n_blocks, np.float32)
        shapes = self.head.noise_shapes()
        tent = None                                                                       # the previous block's continuation (blend > 0)
        for b in range(self.n_blocks):
            t0 = time.perf_counter()
            if self.switch is not None and b == int(self.switch[1]):
                style = str(self.switch[0])
                if 'style_idx' in static:
                    static['style_idx'] = torch.full_like(static['style_idx'], int(STYLE_INDEX[style]))
                if self.text_lookup is not None and token_flags(self.b.module)[0] == 'text':
                    static['text_feat'] = torch.tensor(np.stack([np.asarray(self.text_lookup(c.persona, style), np.float32) for c in cs]), device=dev)
            pivot = hr[:, -1, [0, 2]]                                                     # (B, 2)
            traj_xz, traj_quat = block_plan(xz_m, quat, pivot, b, hop, F)
            root = torch.cat([hr, hr[:, -1:].expand(B, F, 3)], 1).clone()
            root[:, :, [0, 2]] -= pivot[:, None, :]
            raw = dict(static,
                       rotations=torch.cat([hq, hq[:, -1:].expand(B, F, J, 4)], 1),
                       root_pos=root,
                       foot_contact=torch.cat([hc, hc[:, -1:].expand(B, F, 2)], 1),
                       traj_quat=traj_quat.contiguous(), traj_xz=traj_xz.contiguous())
            batch = self.b.module.featurize(raw)
            noise = [draw_noise(s, [noise_seed(c.seed, c.key, b, i) for c in cs]).to(dev) for i, s in enumerate(shapes)]
            sample = self.head.sample(batch, noise)                                       # (B, K+F, J+2, 6)
            fut = sample[:, K:]                                                           # the whole 45-frame future window
            q = repr6d2quat(fut[:, :, :J])
            r = fut[:, :, J, :3] * self.rp_std + self.rp_mean
            r = r.clone(); r[:, :, [0, 2]] += pivot[:, None, :]
            c = fut[:, :, J + 1, :2]
            if self.blend > 0 and tent is not None:
                # realtime/model_loop.py::ChunkStreamer.Poll ("Biped's Previous/Sequence blend"): ease from the previous
                # block's uncommitted continuation into the new block over `blend` frames, weights 1/(n+1) .. n/(n+1);
                # the continuation starts hop frames after the previous block's start, i.e. exactly where this block starts
                tq, tr, tc = tent
                n = min(self.blend, tq.shape[1], q.shape[1])
                w = blend_weights(n, dev, q.dtype, self.blend_shape)
                q = q.clone(); r = r.clone(); c = c.clone()
                q[:, :n] = quat_nlerp_t(tq[:, :n], q[:, :n], w.view(1, n, 1, 1))
                r[:, :n] = (1 - w.view(1, n, 1)) * tr[:, :n] + w.view(1, n, 1) * r[:, :n]
                c[:, :n] = (1 - w.view(1, n, 1)) * tc[:, :n] + w.view(1, n, 1) * c[:, :n]
            tent = (q[:, hop:], r[:, hop:], c[:, hop:]) if self.blend > 0 else None
            q, r, c = q[:, :hop], r[:, :hop], c[:, :hop]
            out_q[:, b * hop:(b + 1) * hop] = q; out_r[:, b * hop:(b + 1) * hop] = r; out_c[:, b * hop:(b + 1) * hop] = c
            hq = torch.cat([hq, q], 1)[:, -K:]; hr = torch.cat([hr, r], 1)[:, -K:]; hc = torch.cat([hc, c], 1)[:, -K:]
            if dev != 'cpu':
                torch.cuda.synchronize()
            block_ms[b] = (time.perf_counter() - t0) * 1000.0
            if self.log and (b % 100 == 0 or b == self.n_blocks - 1):
                self.log(f'  batch {batch_index} ({B0} cases) block {b + 1}/{self.n_blocks}  {block_ms[max(0, b - 99):b + 1].mean():.1f} ms/block')
        out_q, out_r, out_c = out_q.cpu().numpy(), out_r.cpu().numpy(), out_c.cpu().numpy()
        for i in range(B0):
            s = scaled[i]; z = canon[i]
            rows = slice(1, 1 + self.n_frames)
            yield CaseResult(case=cs[i], quats=out_q[i], root=out_r[i], contact=out_c[i], skel_offset=conds[i]['skel'],
                             betas=conds[i]['betas'], leg=legs[i], speed_scale=s.scale, v_cap_leg=s.v_cap_leg,
                             cmd_xz_m=s.xz_m[rows], cmd_speed_mps=s.speed_mps[rows], cmd_heading=s.heading[rows],
                             cmd_facing=s.facing[rows],
                             canonical={k: z[k] for k in ('take', 'start', 'end', 'speed_leg', 'sha256', 'path')},
                             block_ms=block_ms.copy(), batch_size=self.batch_size, batch_index=batch_index)


def blend_weights(n, device, dtype, shape='linear'):
    """Crossfade weights for `n` frames.

    `linear` reproduces realtime/model_loop.py::ChunkStreamer exactly -- weights 1/(n+1) .. n/(n+1) -- and is the
    default so every archived rollout is bit-reproducible. It is, however, only C0: a linear crossfade of two
    POSITION signals leaves a velocity step at each end of the blend window, which is visible in the production
    rollouts as a secondary jerk peak at hop phase 3 (1.37 against a 0.8 interior baseline). `smoothstep` uses
    3w^2 - 2w^3 of the same weights, whose derivative vanishes at both ends, so the velocity is continuous across
    the seam. Costs nothing; must be measured on a codec from the current wave before it becomes the default.
    """
    w = torch.arange(1, n + 1, device=device, dtype=dtype) / (n + 1)
    if shape == 'smoothstep':
        return w * w * (3.0 - 2.0 * w)
    if shape != 'linear':
        raise ValueError(f'blend shape: linear | smoothstep (got {shape!r})')
    return w


def quat_nlerp_t(a, b, w):
    """Normalised lerp with hemisphere alignment (torch), the counterpart of realtime/model_loop.py::quat_nlerp."""
    sign = torch.where((a * b).sum(-1, keepdim=True) < 0, -torch.ones_like(a[..., :1]), torch.ones_like(a[..., :1]))
    out = (1 - w) * a + w * (b * sign)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def facing_of(quats, skel):
    """Actual facing yaw per frame (rad, +Z = 0) from the root orientation: the data's hip-forward definition."""
    fwd = hip_forward(quats[:, 0], skel)
    return np.arctan2(fwd[:, 0], fwd[:, 2])
