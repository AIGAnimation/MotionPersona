# Example data

`sample10/` is a 10-motion subset of the training data, stored in exactly the on-disk format the loader reads
(`data/loco_dataset.py`, layout `flat_v1`); the full dataset has 328,960 motions in the same layout.
`sample10_bvh/` has the same clips as BVH files for viewing (30 fps, centimetres; frame 0 omitted, see below).

| # | performer | style | body | body kind | frames |
|---|---|---|---|---|---|
| 0 | p36 | neutral | p36 | own body | 521 |
| 1 | p05 | angry | p05 | own body | 456 |
| 2 | p24 | happy | p24 | own body | 810 |
| 3 | p06 | depressed | p06 | own body | 745 |
| 4 | p17 | fear | b27 | another captured body | 859 |
| 5 | p29 | drunk | p34 | another captured body | 791 |
| 6 | p14 | bigstep | p36 | another captured body | 702 |
| 7 | p02 | swimming | grid_h02_g02 | synthetic body | 373 |
| 8 | p38 | twofootjump | grid_h00_g04 | synthetic body | 559 |
| 9 | p11 | neutral | grid_h03_g00 | synthetic body | 846 |

"Own body" clips are the captured performance on the performer's own SMPL-X skeleton; the others are the same kind of
take retargeted onto another body by our offline retargeter (the performer, i.e. the persona, is the source; the body
is the target). All 10 clips come from the training part of the split (no held-out body, take or persona x body cell).

## Format

```
sample10/
├── meta.pkl                   skeleton template (T_pose, parents, joint names), root_pos mean/std, counts, layout tag
├── meta.jsonl                 one line per motion: idx, n_frames, frame_offset, source file, target body (skel_name),
│                              text_raw (3 prompt variants), label (style, role, affiliation, dominance, target body)
├── manifest.csv               idx, subject (source performer), stem (source take), style, body id, body name, kind
├── frames/                    all motions concatenated along time (N = 6,662 frames)
│   ├── rotations.npy          (N, 23, 4) float32   local joint rotations, quaternion wxyz (root = world orientation)
│   ├── root_pos.npy           (N, 3)     float32   root (pelvis) position, metres, Y-up
│   ├── foot_contact.npy       (N, 2)     float32   left / right foot contact labels
│   └── traj_pose.npy          (N, 2, 4)  float32   facing orientation, 2 smoothing variants (trajectory conditioning)
└── motions/
    ├── frame_offsets.npy      (M+1,)     int64     motion i owns frames [offsets[i], offsets[i+1])
    ├── shape_feat.npy         (M, 10)    float32   SMPL-X betas of the target body
    ├── skel_offset.npy        (M, 23, 3) float32   parent-relative joint offsets (metres); row 0 = pelvis above the sole plane
    ├── text_feat.npy          (M, 3, 512) float32  CLIP features of the 3 persona + style prompts (not used by the released prior)
    └── persona_feat.npy       (M, 3, 512) float32  CLIP features of persona-only prompts (prompt-conditioned ablation only)
```

The 23 joints are the SMPL-X body joints: pelvis, left/right hip, knee, ankle, foot, spine1-3, neck, head, jaw,
left/right collar, shoulder, elbow, wrist. Frame 0 of every clip is the reference pose of the source capture; the
loader uses the data as stored, the BVH exports start at frame 1.

Reading one motion:

```python
import numpy as np
d = 'example_data/sample10'
off = np.load(f'{d}/motions/frame_offsets.npy')
rot = np.load(f'{d}/frames/rotations.npy', mmap_mode='r')
i = 3
quats = rot[off[i]:off[i + 1]]            # (T, 23, 4)
```

Training on it: `python train.py experiment=example_codec` (see the main README).
