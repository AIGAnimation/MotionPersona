"""Model-in-the-loop for the realtime scene: chunk-by-chunk generation with the trained persona heads.

ai4animationpy Biped-demo structure (Control -> Predict -> Animate), but the network is a chunk model: it takes the
last K=10 frames of the character and the 45-frame trajectory plan from the current root, and returns the next
45 frames at once. So instead of re-predicting the whole future window every 1/PREDICTION_FPS,
we keep a committed frame stream that the display plays back at 30 fps and ask the model for the next chunk
whenever the remaining buffer drops below `lead` frames; the request runs in a background thread so the render
loop never blocks (torch releases the GIL).

Frame conventions (data/loco_dataset.py get_raw_item + data/augment.py featurize_batch, world frame, theta = 0):
  rotations (T,J=23,4) local quats wxyz, root = world orientation; root_pos (T,3) metres, xz relative to the pivot
  frame K-1, y absolute; foot_contact (T,2); traj_xz (45,2) / traj_quat (45,4) for frames K..K+44 relative to
  the pivot; aug_trig = [1,0,1,0]; shape_feat = betas; skel_offset = data.bodies.betas_to_offsets(betas,
  root_basis=<the ckpt's data.skel_root_basis, 'joint' when absent>) -- y = 0 is the sole plane (sole basis) or
  the lowest foot joint (joint basis) and the root height follows that basis.
  The runtime world is the same right-handed Y-up frame as the BVH world (+Z = hip forward at rest), so no
  rotation is applied on either side: conditioning and history are fed in world axes.

    model = PersonaModel('checkpoints/prior.ckpt', steps=2)
    cond = model.Conditions('p02', 'neutral')          # CLIP row, or the learned persona/style rows -- see below
    quats, root, contact = model.generate(hist_quats, hist_root, hist_contact, traj_xz, traj_quat, betas, cond)

Persona conditions: a checkpoint trained with `model.persona_cond=id_attr` (the released prior) takes the persona as
a performer-ID row plus three typed attribute rows (and its style as a row of STYLE_VOCAB) instead of a CLIP feature.
`Conditions(subject, style)` returns whatever THIS checkpoint needs, reading assets/persona_rows.json; a bare (512,)
array is still accepted for CLIP-token checkpoints (persona_cond=text).
"""
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf                                            # noqa: E402

from data.bodies import REPO_NAMES, REPO_PARENTS, betas_to_offsets, load_regressor   # noqa: E402
from data.loco_dataset import STYLE_INDEX, STYLE_VOCAB                       # noqa: E402
from network.checkpoint import ema_merged, extract_model_state, fill_released                         # noqa: E402
from network.factory import build_module                                   # noqa: E402
from utils import nn_transforms                                            # noqa: E402

FPS = 30
FM_STEPS = 2                      # flow-matching heads: 1-2 euler steps beat the trained 8
THREADS = 2                       # the heads are tiny: 1-4 torch threads = 17 ms/chunk, the default 14 = 27 ms and starves the render loop
TEXT_FEATS = Path(__file__).resolve().parent / 'assets/text_feats.npz'
PERSONA_ROWS = Path(__file__).resolve().parent / 'assets/persona_rows.json'
TEXT_DIM = 512                    # CLIP feature width (only needed for the --compile warm-up input)
REGRESSOR = REPO / 'data/bodies/joint_regressor.npz'
L_HIP, R_HIP = REPO_NAMES.index('left_hip'), REPO_NAMES.index('right_hip')


# ----------------------------------------------------------------------------- numpy rotation helpers (wxyz)
def quat_to_mat(q):
    """(...,4) wxyz -> (...,3,3), same formula as utils.nn_transforms.quat2mat."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    m = np.empty(q.shape[:-1] + (3, 3), q.dtype)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z); m[..., 0, 1] = 2 * (x * y - z * w); m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w); m[..., 1, 1] = 1 - 2 * (x * x + z * z); m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w); m[..., 2, 1] = 2 * (y * z + x * w); m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def quat_nlerp(a, b, w):
    """Normalised lerp with hemisphere alignment, (...,4) each, w scalar or broadcastable."""
    b = np.where(np.sum(a * b, -1, keepdims=True) < 0, -b, b)
    q = (1 - w) * a + w * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def fk(quats, root, offsets, parents):
    """Forward kinematics like utils.nn_transforms.neural_FK: quats (T,J,4) local wxyz (root = world), root (T,3),
    offsets (J,3) parent-relative, parents list -> world positions (T,J,3), world rotations (T,J,3,3)."""
    T, J = quats.shape[:2]
    rot = quat_to_mat(np.asarray(quats, np.float64))
    pos = np.zeros((T, J, 3))
    pos[:, 0] = root
    for j, p in enumerate(parents):
        if p < 0:
            continue
        pos[:, j] = pos[:, p] + np.einsum('tij,j->ti', rot[:, p], offsets[j])
        rot[:, j] = rot[:, p] @ rot[:, j]
    return pos, rot


def hip_forward(root_quat, offsets):
    """Data's facing definition (utils.motion_processing.extract_forward_hips): cross(left_hip - right_hip, +Y),
    which only depends on the root orientation and the hip offsets. root_quat (...,4) -> (...,3) unit vectors."""
    across = quat_to_mat(np.asarray(root_quat, np.float64)) @ (offsets[L_HIP] - offsets[R_HIP])
    fwd = np.cross(across, np.array([0.0, 1.0, 0.0]))
    return fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)


def assemble_smplx(pos, rot, joints, parents, driven, lift=0.0):
    """World 4x4 transforms for the 55 SMPL-X joints from the 23 driven joints' FK (pos (23,3), rot (23,3,3)):
    driven joints take the FK result (+ a vertical display lift), the others (hands, eyes) hang off their parent
    with the rest offset from `joints` (55,3) and no rotation of their own. `driven[k]` = SMPL-X index of repo joint k."""
    n = len(joints)
    T = np.tile(np.eye(4), (n, 1, 1))
    repo = {int(j): k for k, j in enumerate(driven)}
    for j in range(n):
        k = repo.get(j)
        if k is not None:
            T[j, :3, :3] = rot[k]
            T[j, :3, 3] = pos[k]
            T[j, 1, 3] += lift
        else:
            p = int(parents[j])
            T[j, :3, :3] = T[p, :3, :3]
            T[j, :3, 3] = T[p, :3, 3] + T[p, :3, :3] @ (joints[j] - joints[p])
    return T


# ----------------------------------------------------------------------------- persona rows / text prompts
class PersonaRows:
    """assets/persona_rows.json: performer -> ID row of SUBJECT_VOCAB and the three
    typed attribute rows, i.e. what a `persona_cond=id_attr` checkpoint takes instead of a CLIP prompt feature."""

    def __init__(self, path=PERSONA_ROWS):
        if not Path(path).exists():
            raise SystemExit(f'{path} missing: it ships with the repository (needed by checkpoints '
                             'whose persona enters as learned tokens)')
        doc = json.load(open(path))
        self.Subjects = sorted(doc['subjects'])
        self._rows = doc['subjects']
        self.AttrKeys = list(doc['attr_keys'])

    def Get(self, subject):
        """-> (subject row, (3,) attribute rows) for one performer."""
        r = self._rows.get(str(subject).strip().lower())
        if r is None:
            raise SystemExit(f'unknown performer {subject!r}; assets/persona_rows.json has {len(self.Subjects)}')
        return np.int64(r['id']), np.asarray(r['attr'], np.int64)

    def Describe(self, subject):
        r = self._rows[str(subject).strip().lower()]
        return ', '.join(f'{k} {r[k]}' for k in self.AttrKeys)


class TextFeats:
    """CLIP features of the training prompts (assets/text_feats.npz, dumped from the dataset meta):
    one row per (subject, style, variant), variant 0..2 = the three prompt phrasings of each motion."""

    def __init__(self, path=TEXT_FEATS):
        z = np.load(path, allow_pickle=True)
        self.Subject = [str(s) for s in z['subject']]
        self.Style = [str(s) for s in z['style']]
        self.Variant = np.asarray(z['variant'], np.int64)
        self.Prompt = [str(p) for p in z['prompt']]
        self.Feat = np.asarray(z['feat'], np.float32)
        self.Subjects = sorted(set(self.Subject))
        self._index = {(s, st, int(v)): i for i, (s, st, v) in enumerate(zip(self.Subject, self.Style, self.Variant))}

    def Styles(self, subject):
        return sorted({st for s, st in zip(self.Subject, self.Style) if s == subject})

    def Get(self, subject, style, variant=0):
        i = self._index.get((subject, style, int(variant)))
        if i is None:
            i = self._index[(subject, style, 0)]
        return self.Feat[i], self.Prompt[i]


# ----------------------------------------------------------------------------- the network
class PersonaModel:
    """A trained head (ddpm / fm / latfm checkpoint) wrapped for one-chunk generation on the CPU."""

    def __init__(self, ckpt, device='cpu', steps=None, ema=False, threads=None, compile=False):
        ckpt = Path(ckpt)
        if not ckpt.exists() and not ckpt.is_absolute():
            ckpt = REPO / ckpt                                                  # 'save/<run>/best.ckpt' from any cwd
        self.Path = ckpt.resolve()
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        cfg = OmegaConf.create(state['hyper_parameters']['cfg'])
        meta = state['hyper_parameters']['meta']
        self.Kind = str(cfg.diffusion.get('type', 'ddpm'))
        if self.Kind == 'latfm':
            vae = Path(str(cfg.diffusion.vae_ckpt))
            if not vae.is_absolute():                                          # trained with a cwd-relative path;
                vae = ckpt.parent / vae if (ckpt.parent / vae).exists() else REPO / vae   # released ckpts: codec next to the prior
            cfg.diffusion.vae_ckpt = str(vae)
        torch.set_num_threads(int(THREADS if threads is None else threads))
        module = build_module(cfg, meta)
        sd = extract_model_state(state)
        own = module.model.state_dict()
        fill_released(sd, own)
        if 'clean_token' in own and 'clean_token' not in sd:
            sd['clean_token'] = torch.zeros_like(own['clean_token'])
        module.model.load_state_dict(sd, strict=True)
        if ema and not ema_merged(state):                                   # released ckpts: already EMA
            from torch_ema import ExponentialMovingAverage
            ema_state = state.get('callbacks', {}).get('EMACallback')
            if not ema_state:
                raise SystemExit(f'{ckpt}: no EMA weights stored')
            shadow = ExponentialMovingAverage(module.model.parameters(), decay=ema_state['decay'])
            shadow.load_state_dict(ema_state)
            shadow.copy_to()
        module.eval().to(device)
        self.Module, self.Cfg, self.Device = module, cfg, device
        self.K, self.F = int(cfg.data.past_frame), int(cfg.data.future_frame)
        self.J = int(meta['joint_num'])
        self.Names, self.Parents = list(meta['names']), [int(p) for p in meta['parents']]
        assert self.Names == REPO_NAMES and self.Parents == REPO_PARENTS.tolist(), 'checkpoint skeleton != repo skeleton'
        self.RootMean = np.asarray(meta['root_pos_mean'], np.float32)
        self.RootStd = np.asarray(meta['root_pos_std'], np.float32)
        self.Steps = None
        if self.Kind in ('latfm', 'fm'):
            self.Steps = int(steps) if steps else FM_STEPS
            module.fm_steps = self.Steps                                        # euler 1-2 steps is the sweet spot
        elif steps:
            cfg.diffusion.num_inference_steps = int(steps)
            self.Steps = int(steps)
        else:
            self.Steps = int(cfg.diffusion.num_train_timesteps)
        self.Regressor = load_regressor(REGRESSOR)
        # what skel_offset[0] meant in the training data (configs/data/default.yaml): 'sole' = pelvis above the
        # rigid-foot sole plane, 'joint' = above the lowest foot joint (a ckpt without the key). Feeding the other
        # basis shifts the root input by ~1.5 cm.
        self.RootBasis = str(cfg.data.get('skel_root_basis', 'joint'))
        # how this checkpoint takes the persona and the style (absent keys = the CLIP token)
        self.PersonaCond = str(cfg.model.get('persona_cond', 'text'))
        self.StyleCond = str(cfg.model.get('style_cond', 'none'))
        self.Rows = PersonaRows() if self.PersonaCond in ('id', 'attr', 'id_attr') else None
        self.Name = 'MotionPersona_V1' if ema_merged(state) else ckpt.parent.name   # released ckpt / run dir
        self.LastMs = 0.0
        self.CompileMs = 0.0
        if compile:
            self.Compile()

    def Compile(self):
        """torch.compile the three hot calls (denoiser, VAE decode, history encode). Measured bit-exact (max quat
        deviation 0.0000 over 40 chunks) and worth 9.7 -> 6.0 ms on an RTX 5080 with CUDA graphs; on CPU it buys nothing
        (the work is already BLAS) and macOS inductor is broken, so this is opt-in via --compile. The warm-up (the
        actual compilation, ~6-13 s) is paid here rather than on the first chunk of the demo."""
        mode = 'reduce-overhead' if self.Device == 'cuda' else 'default'         # reduce-overhead = CUDA graphs
        vae = getattr(self.Module, 'vae', None)
        targets = [(self.Module.model, 'forward')] + ([(vae, 'decode'), (vae, 'encode_past')] if vae is not None else [])
        saved = [(obj, name, getattr(obj, name)) for obj, name in targets]
        t0 = time.perf_counter()
        try:
            for obj, name, fn in saved:
                setattr(obj, name, torch.compile(fn, mode=mode, dynamic=False))
            betas = np.zeros(10)
            quats, root, contact = self.RestHistory(betas)
            traj_xz = np.stack([np.zeros(self.F), np.arange(1, self.F + 1) / FPS], -1).astype(np.float32)
            traj_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (self.F, 1))
            warm = ({'text_feat': np.zeros(TEXT_DIM, np.float32)} if self.PersonaCond == 'text'
                    else self.Conditions(self.Rows.Subjects[0], 'neutral'))      # shapes are what compilation needs
            for _ in range(3):
                self.generate(quats, root, contact, traj_xz, traj_quat, betas, warm)
        except Exception as exc:                                                 # inductor is fragile; never fatal
            for obj, name, fn in saved:
                setattr(obj, name, fn)
            print(f'--compile disabled ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})')
            return
        self.CompileMs = (time.perf_counter() - t0) * 1000.0

    def Conditions(self, subject, style, variant=0, texts=None):
        """The conditions THIS checkpoint needs for one (performer, style), as a dict for `generate`:
        a CLIP row for persona_cond=text (from assets/text_feats.npz, or `texts` if you already loaded it), and/or
        the learned rows (subject / attributes / style) for the id_attr family."""
        cond = {}
        if self.PersonaCond == 'text':
            feats = texts if texts is not None else TextFeats()
            cond['text_feat'] = np.asarray(feats.Get(subject, style, variant)[0], np.float32)
        elif self.Rows is not None:
            sid, attr = self.Rows.Get(subject)
            if self.Rows and self.PersonaCond != 'attr':
                cond['subject_idx'] = sid
            if self.PersonaCond != 'id':
                cond['attr_idx'] = attr
        if self.StyleCond == 'embed':
            if str(style) not in STYLE_INDEX:
                raise SystemExit(f'unknown style {style!r}; STYLE_VOCAB = {STYLE_VOCAB}')
            cond['style_idx'] = np.int64(STYLE_INDEX[str(style)])
        return cond

    def AcceptsStyle(self, style):
        """Can this checkpoint be asked for that style? A CLIP-token checkpoint takes any prompt spelling, but once
        the style is a learned token it must be a row of STYLE_VOCAB -- assets/text_feats.npz also carries the raw
        label spellings ('angry_id', '_neutral', 'twofootump', ...), which have no embedding row."""
        return self.StyleCond != 'embed' or str(style) in STYLE_INDEX

    def Offsets(self, betas):
        """skel_offset for these betas (J,3) metres; the root row is in the checkpoint's basis (self.RootBasis)."""
        return betas_to_offsets(np.asarray(betas, np.float64).reshape(1, -1), self.Regressor, root_basis=self.RootBasis)[0]

    def RestHistory(self, betas):
        """K frames of the standing rest pose at the origin facing +Z: the cold-start history."""
        off = self.Offsets(betas)
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (self.K, self.J, 1))
        root = np.tile(np.array([0.0, off[0, 1], 0.0], np.float32), (self.K, 1))
        contact = np.ones((self.K, 2), np.float32)
        return quats, root, contact

    @torch.no_grad()
    def generate(self, hist_quats, hist_root, hist_contact, traj_xz, traj_quat, betas, cond, generator=None,
                 guidance=1.0):
        """One chunk. hist_* = the last K frames in world axes (quats (K,J,4), root (K,3) m, contact (K,2)); traj_xz (F,2)
        / traj_quat (F,4) = the plan for the F future frames relative to the pivot (frame K-1) in world axes.
        `cond` = the conditions dict from `Conditions()`, or a bare (512,) CLIP feature (CLIP-token checkpoints).
        Returns quats (F,J,4), root (F,3) world, contact (F,2)."""
        K, F, J = self.K, self.F, self.J
        assert hist_quats.shape == (K, J, 4) and hist_root.shape == (K, 3) and traj_xz.shape == (F, 2), \
            (hist_quats.shape, hist_root.shape, traj_xz.shape)
        pivot = np.asarray(hist_root[-1, [0, 2]], np.float32).copy()
        rot = np.concatenate([hist_quats, np.repeat(hist_quats[-1:], F, 0)], 0).astype(np.float32)
        root = np.concatenate([hist_root, np.repeat(hist_root[-1:], F, 0)], 0).astype(np.float32)
        root[:, [0, 2]] -= pivot
        contact = np.concatenate([hist_contact, np.repeat(hist_contact[-1:], F, 0)], 0).astype(np.float32)
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a))[None]
        raw = {
            'rotations': t(rot), 'root_pos': t(root), 'foot_contact': t(contact),
            'traj_quat': t(np.asarray(traj_quat, np.float32)), 'traj_xz': t(np.asarray(traj_xz, np.float32)),
            'aug_trig': t(np.array([1.0, 0.0, 1.0, 0.0], np.float32)),
            'shape_feat': t(np.asarray(betas, np.float32)),
            'skel_offset': t(self.Offsets(betas)),
            'motion_idx': torch.zeros(1, dtype=torch.int64), 'clip_start': torch.zeros(1, dtype=torch.int64),
        }
        cond = {'text_feat': cond} if not isinstance(cond, dict) else cond      # bare CLIP row = persona_cond=text
        for k in ('text_feat', 'subject_idx', 'attr_idx', 'style_idx'):
            if k in cond:
                v = np.asarray(cond[k], np.float32 if k == 'text_feat' else np.int64)
                # The scalar rows (subject / style) have to arrive as (B,), the way the collated training batch has
                # them: nn.Embedding indexes them directly, so a stray axis makes a 4-D token that only blows up
                # later in the denoiser's torch.cat. (`t()` cannot be used -- np.ascontiguousarray turns a 0-d
                # array into shape (1,), and the [None] then makes it (1,1).)
                raw[k] = torch.as_tensor(v.reshape(1)) if v.ndim == 0 else t(v)
        if self.PersonaCond == 'text' and 'text_feat' not in raw:
            raise SystemExit('this checkpoint takes the CLIP persona token: pass Conditions(subject, style) or a (512,) feature')
        for k in (['subject_idx'] if self.PersonaCond in ('id', 'id_attr') else []) + \
                 (['attr_idx'] if self.PersonaCond in ('attr', 'id_attr') else []) + \
                 (['style_idx'] if self.StyleCond == 'embed' else []):
            if k not in raw:
                raise SystemExit(f'this checkpoint needs {k}: build the conditions with Conditions(subject, style)')
        if 'text_feat' not in raw:
            # data/augment.py::featurize_batch reads raw['text_feat'] unconditionally (every training batch carries
            # the CLIP row whether or not the head uses it); a learned-persona prior drops it in _text(), so any
            # value does -- but the key has to exist.
            raw['text_feat'] = t(np.zeros(TEXT_DIM, np.float32))
        t0 = time.perf_counter()
        self.Module.hist_guidance = float(guidance)                           # CFG on the history token (1 = off)
        try:
            batch = self.Module.featurize(raw)
            _, sample = self.Module.sample(batch, generator)                  # (1, K+F, J+2, 6)
        finally:
            self.Module.hist_guidance = 1.0
        fut = sample[0, K:].cpu()
        quats = nn_transforms.repr6d2quat(fut[:, :J]).numpy().astype(np.float32)
        root_out = fut[:, J, :3].numpy() * self.RootStd + self.RootMean
        root_out[:, [0, 2]] += pivot
        contact_out = fut[:, J + 1, :2].numpy()
        self.LastMs = (time.perf_counter() - t0) * 1000.0
        return quats, root_out.astype(np.float32), contact_out.astype(np.float32)


# ----------------------------------------------------------------------------- the stream
class ChunkStreamer:
    """Committed frame stream played at 30 fps + asynchronous chunk requests.

    `hop` = frames committed from every generated chunk (45 = strict chunk-by-chunk; smaller = overlapping
    regeneration, Biped-like); the remaining frames of the chunk stay 'tentative' — the model's own continuation,
    used as the trajectory-correction target and crossfaded (`blend` frames) with the next chunk. `lead` = the
    buffer level (frames ahead of the playhead) at which the next request is launched; it must cover the
    inference latency (a stall freezes the character on the last frame and is counted in `Stalls`).

    The committed frames ahead of the playhead (between `lead` and `lead + hop`) were generated from an older plan,
    so they are pure input dead time: hop 15 / lead 8 gave idle->run t50 0.83 s and a 90-degree turn t90 1.5 s;
    hop 3 / lead 3 / blend 2 (a chunk every 100 ms, ~20-35 ms each on 2 CPU threads) gives 0.28 s / 0.9 s with the
    seams as smooth as a normal step. Keep blend < hop, otherwise no committed frame is ever fully the new chunk."""

    def __init__(self, model, hop=3, lead=3, blend=2, seed=None, hop_steady=None, guidance=0.5, boost_chunks=8):
        self.Model = model
        self.HopBase = int(np.clip(hop, 1, model.F))                            # while the input is changing
        self.HopSteady = int(np.clip(hop if hop_steady is None else hop_steady, self.HopBase, model.F))
        self.Hop = self.HopBase
        self.Lead = int(lead)
        self.Blend = int(blend)
        self.Generator = torch.Generator(device=model.Device).manual_seed(int(seed)) if seed is not None else None
        self.Offsets = None
        self._lock = threading.Lock()
        self._job = None
        self._queue = queue.Queue()
        self._worker = None
        self.Guidance = float(guidance)                                         # history-CFG weight while boosting
        self.BoostChunks = int(boost_chunks)
        self.BlendNext = 0          # Transition(): a longer, smoothstep cross-fade for the next commit(s) (0 = off)
        self.BoostLeft = 0
        self.Stalls = 0
        self.Chunks = 0
        self.LatencyMs = 0.0
        self.ComputeMs = 0.0                                                    # generate() time of the last committed chunk
        self.ComputeCpuMs = 0.0                                                 # ... the worker thread's CPU time of it
        self.TimingRepeat = 1                                                   # >1: also time each chunk min-of-N (_retime)
        self.Error = None
        # Stream bookkeeping for the recorder (stream_io.StreamRecorder), inert otherwise: `Base` = stream index of
        # Quats[0] (the lists are trimmed at the front), `FrameBlock` = which chunk produced each committed frame
        # (-1 = the history the stream was reset with), `OnCommit(info)` = called once per committed chunk.
        self.Base = 0
        self.FrameBlock = []
        self.Launched = 0                                                       # chunks launched since the last Reset
        self.OnCommit = None

    # ----------------------------------------------------------------- state
    def Reset(self, betas, quats=None, root=None, contact=None):
        """Start from a history (default: the rest pose of this body at the origin)."""
        self.Offsets = self.Model.Offsets(betas)
        if quats is None:
            quats, root, contact = self.Model.RestHistory(betas)
        self.Quats = [q for q in np.asarray(quats, np.float32)]
        self.Root = [r for r in np.asarray(root, np.float32)]
        self.Contact = [c for c in np.asarray(contact, np.float32)]
        self.Tentative = None                                                   # (quats, root, contact) beyond the commit
        self.T = float(len(self.Quats) - 1)                                     # playhead, frame units
        self._job = None
        self.Base = 0
        self.FrameBlock = [-1] * len(self.Quats)
        self.Launched = 0

    def _forward(self, quats):
        return hip_forward(np.asarray(quats)[..., 0, :], self.Offsets)

    def Remaining(self):
        """Committed frames ahead of the playhead."""
        return len(self.Quats) - 1 - int(np.floor(self.T))

    def Busy(self):
        return self._job is not None

    # ----------------------------------------------------------------- playback
    def Advance(self, dt):
        self.T += dt * FPS
        last = float(len(self.Quats) - 1)
        if self.T > last:
            self.T = last
            self.Stalls += int(self.Chunks > 0)                                 # the cold start waits for chunk 1 by design
        # trim the past (keep K frames of history behind the playhead plus some slack)
        drop = int(np.floor(self.T)) - self.Model.K - 30
        if drop > 0:
            del self.Quats[:drop]; del self.Root[:drop]; del self.Contact[:drop]; del self.FrameBlock[:drop]
            self.T -= drop
            self.Base += drop

    def Boost(self, chunks=None, scale=None):
        """Generate the next few chunks with the history's grip loosened (MotionDiffusionModule.guided_by_history).

        The demo calls this when the prompt changes: measured on a full-data prior, switching p03 neutral -> bigstep
        while running reaches only 44 % of the target stride after 5 chunks at w = 1, but 68 % at w = 0.5 -- the
        model keeps reproducing the old style out of its own history. Costs a second forward per denoising step,
        so it is a short window, not a permanent setting."""
        self.BoostLeft = int(self.BoostChunks if chunks is None else chunks)
        if scale is not None:
            self.Guidance = float(scale)

    def Boosting(self):
        return self.BoostLeft > 0

    def Transition(self, frames):
        """A prompt change (style / subject) is coming: cross-fade the next chunk into the old stream's continuation over
        `frames` frames (smoothstep) instead of `Blend`, carrying what one commit does not cover into the next chunk.
        Pair with Invalidate(keep_tentative=True), which keeps the dropped frames as that continuation. Off (0) by
        default; without it a prompt change cut straight to the new chunk: Invalidate() cleared `Tentative`, so not even
        the 2-frame blend ran and the window-start jerk the blend exists for hit the stream (measured on a scripted
        run: max joint accel 6.2 / 13.7 / 3.1 cm/frame^2 at three style switches vs a 1.45 median; other block joins
        1.3-1.7)."""
        self.BlendNext = max(int(frames), 0)

    def Invalidate(self, keep=None, keep_tentative=False):
        """The input changed under a long hop: drop committed frames further than `keep` ahead of the playhead (they
        were generated from a plan the user has since abandoned — pure input dead time) and mark the chunk in flight
        stale, since its history is the frames we just dropped. Returns the number of frames dropped.

        The stale chunk is dropped at Poll rather than here, so two chunks never run at once. Callers only invalidate
        when there is dead time beyond `Lead + HopBase` to reclaim, so the responsive path never discards work."""
        keep = (self.Lead + self.HopBase) if keep is None else int(keep)
        drop = self.Remaining() - keep
        if drop > 0:
            if keep_tentative:                                                   # the dropped frames ARE the old
                tq = list(self.Quats[-drop:]); tr = list(self.Root[-drop:]); tc = list(self.Contact[-drop:])   # continuation
                if self.Tentative is not None:
                    tq += list(self.Tentative[0]); tr += list(self.Tentative[1]); tc += list(self.Tentative[2])
            del self.Quats[-drop:]; del self.Root[-drop:]; del self.Contact[-drop:]; del self.FrameBlock[-drop:]
            self.Tentative = (np.asarray(tq), np.asarray(tr), np.asarray(tc)) if keep_tentative else None
        if self._job is not None:
            self._job['stale'] = True
        self.Hop = self.HopBase
        return max(drop, 0)

    def Sample(self):
        """Pose at the playhead: quats (J,4), root (3), forward (3), contact (2)."""
        i = int(np.floor(self.T))
        w = self.T - i
        j = min(i + 1, len(self.Quats) - 1)
        quats = quat_nlerp(self.Quats[i], self.Quats[j], w) if w > 0 else self.Quats[i]
        root = (1 - w) * self.Root[i] + w * self.Root[j]
        contact = (1 - w) * self.Contact[i] + w * self.Contact[j]
        return quats, root, self._forward(quats), contact

    def FuturePath(self, n):
        """Root positions / forwards for the frames floor(T) .. floor(T)+n-1 that the stream already knows (committed
        then tentative); returns fewer than n rows when the buffer is shorter."""
        i = int(np.floor(self.T))
        quats = self.Quats[i:i + n]
        root = self.Root[i:i + n]
        if self.Tentative is not None and len(quats) < n:
            tq, tr, _ = self.Tentative
            k = n - len(quats)
            quats = quats + [q for q in tq[:k]]
            root = root + [r for r in tr[:k]]
        quats = np.asarray(quats, np.float32)
        return np.asarray(root, np.float64), self._forward(quats)

    # ----------------------------------------------------------------- generation
    def Launch(self, traj_xz, traj_quat, betas, text_feat, tag=None):
        """Request the next chunk from the last K committed frames (pivot = the last committed frame).
        `text_feat` is the conditions dict of PersonaModel.Conditions(), or a bare CLIP row (CLIP-token checkpoints).
        `tag` (optional, e.g. the subject / style names) rides along to `OnCommit` untouched.

        Every window starts with a jerk spike where the prediction joins the history (toe jerk at window frames 0-2
        is 4.2-4.7x the mid-window level, across codec variants), and hop 3 commits exactly those frames. Requesting the
        window from a few frames BEHIND the committed end and skipping its first frames was tried and
        measured worse either way: dropping the regenerated frames feeds a forward offset back through the history
        (root speed ran away), cross-fading them into the committed buffer doubles the toe jerk (0.68 -> 1.49 at
        hop 3). The `blend` cross-fade into the previous chunk's continuation below is what handles the junction."""
        assert self._job is None
        K = self.Model.K
        hist = (np.asarray(self.Quats[-K:], np.float32), np.asarray(self.Root[-K:], np.float32),
                np.asarray(self.Contact[-K:], np.float32))
        guidance = self.Guidance if self.BoostLeft > 0 else 1.0
        self.BoostLeft = max(self.BoostLeft - 1, 0)
        cond = dict(text_feat) if isinstance(text_feat, dict) else np.asarray(text_feat, np.float32).copy()
        args = (hist, np.asarray(traj_xz, np.float32).copy(), np.asarray(traj_quat, np.float32).copy(),
                np.asarray(betas, np.float32).copy(), cond, guidance)
        job = {'result': None, 'error': None, 't0': time.perf_counter(), 'stale': False, 'done': False,
               'block': self.Launched, 'betas': args[3], 'guidance': guidance, 'tag': tag, 'ms': 0.0, 'cpu_ms': 0.0,
               'event': threading.Event()}
        self.Launched += 1
        self._job = job
        if self._worker is None:                                                 # one long-lived worker, not one
            self._worker = threading.Thread(target=self._serve, daemon=True)     # thread per chunk: torch keeps its
            self._worker.start()                                                 # per-thread state (and CUDA graphs
        self._queue.put((job, args))                                             # only replay on the capturing thread)

    def _serve(self):
        while True:
            job, args = self._queue.get()
            t0, c0 = time.perf_counter(), time.thread_time()
            try:
                job['result'] = self.Model.generate(*args[0], *args[1:-1], generator=self.Generator,
                                                    guidance=args[-1])
            except Exception as exc:                                             # surfaced in the HUD, not lost in the thread
                job['error'] = exc
            # this chunk's time, queueing excluded: wall (what the budget sees, GIL waits behind the render loop
            # included) and the worker thread's own CPU time (the cost of the chunk itself)
            job['ms'] = (time.perf_counter() - t0) * 1000.0
            job['cpu_ms'] = (time.thread_time() - c0) * 1000.0
            job['min_ms'] = job['ms']
            if self.TimingRepeat > 1 and job['error'] is None:
                job['min_ms'] = self._retime(args, job['ms'])
            job['done'] = True
            job['event'].set()

    def _retime(self, args, first_ms):
        """--timing-repeat N (offline drivers only): run the same chunk N-1 more times with the generator rewound, so
        the stream is untouched, and return the fastest wall time -- a per-chunk cost that survives a machine shared
        with other jobs (e.g. a renderer running next to it)."""
        g = self.Generator
        best = first_ms
        for _ in range(self.TimingRepeat - 1):
            state = g.get_state() if g is not None else None
            t0 = time.perf_counter()
            self.Model.generate(*args[0], *args[1:-1], generator=g, guidance=args[-1])
            best = min(best, (time.perf_counter() - t0) * 1000.0)
            if g is not None:
                g.set_state(state)
        return best

    def Wait(self, timeout=60.0):
        """Block until the chunk in flight has been generated (it is committed by the next Poll). Only the offline
        drivers use this (Program --headless / --capture): a fixed virtual clock with every chunk arriving one render
        frame after its launch, independent of how fast the machine is."""
        job = self._job
        # block on the event, not a sleep-poll: a thread waking every half millisecond keeps asking the chunk thread
        # for the GIL and more than doubled the measured chunk time on CUDA (25-42 ms vs 10 ms standalone)
        if job is not None and not job['event'].wait(timeout):
            raise TimeoutError('chunk did not arrive within %.0f s' % timeout)

    def Poll(self):
        """Commit a finished chunk. Returns True when new frames arrived."""
        job = self._job
        if job is None or not job['done']:
            return False
        self._job = None
        if job.get('stale'):                                                     # Invalidate() dropped its history
            return False
        if job['error'] is not None:
            self.Error = job['error']
            return False
        quats, root, contact = job['result']
        self.LatencyMs = (time.perf_counter() - job['t0']) * 1000.0
        self.ComputeMs = float(job['ms'])
        self.ComputeCpuMs = float(job['cpu_ms'])
        self.Chunks += 1
        trans = self.BlendNext > 0
        if self.Tentative is not None and (self.Blend > 0 or trans):
            # Biped's Previous/Sequence blend: ease from the previous chunk's continuation into the new one
            tq, tr, tc = self.Tentative
            n = min(self.BlendNext if trans else self.Blend, len(tq), len(quats))
            w = (np.arange(1, n + 1) / (n + 1)).astype(np.float32)
            if trans:                                                            # prompt change: smoothstep ramp
                w = (w * w * (3.0 - 2.0 * w)).astype(np.float32)
            quats[:n] = quat_nlerp(tq[:n], quats[:n], w[:, None, None])
            root[:n] = (1 - w[:, None]) * tr[:n] + w[:, None] * root[:n]
            contact[:n] = (1 - w[:, None]) * tc[:n] + w[:, None] * contact[:n]
        h = self.Hop
        if trans:                                                                # carry the rest of a long transition
            self.BlendNext = max(self.BlendNext - h, 0)
            if 0 < self.BlendNext <= self.Blend:
                self.BlendNext = 0
        start = self.Base + len(self.Quats)                                     # stream index of the first new frame
        self.Quats += [q for q in quats[:h]]
        self.Root += [r for r in root[:h]]
        self.Contact += [c for c in contact[:h]]
        self.FrameBlock += [job['block']] * len(quats[:h])
        self.Tentative = (quats[h:], root[h:], contact[h:]) if h < len(quats) else None
        if self.OnCommit is not None:
            self.OnCommit({'block': job['block'], 'start': start, 'n': len(quats[:h]), 'hop': h, 'ms': self.ComputeMs,
                           'cpu_ms': self.ComputeCpuMs, 'min_ms': float(job.get('min_ms', self.ComputeMs)),
                           'latency_ms': self.LatencyMs, 'betas': job['betas'], 'guidance': job['guidance'],
                           'tag': job['tag']})
        return True
