"""Quick look at generated motion: BVH files -> a stick-figure animation (mp4 or gif), side by side.

    python scripts/visualize.py outputs/p02__p02__happy__s0.bvh                          # -> outputs/p02__p02__happy__s0.mp4
    python scripts/visualize.py outputs/a.bvh outputs/b.bvh -o compare.gif --seconds 10   # two panels, first 10 s
    python scripts/visualize.py outputs/*.bvh --overview                                  # whole route in view

Works on the BVH files of generate.py (and on any BVH of the repository's SMPL-X body skeleton). When the matching
`<stem>.npz` of generate.py is next to a BVH, the commanded route is drawn on the ground (blue) together with the
root path (orange) and the predicted foot contacts (red feet). For mesh renders, import the BVH into Blender
(File > Import > Motion Capture (.bvh), scale 0.01: the files are in centimetres) and drive an SMPL-X body with it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.bvh_motion import Motion  # noqa: E402

FEET = ('left_foot', 'right_foot')


def load(path):
    m = Motion.load_bvh(str(path))
    m.update_global_positions()
    pos = np.asarray(m.global_positions, np.float64) / 100.0             # cm -> m, (T, J, 3), y-up
    clip = dict(name=Path(path).stem, pos=pos, parents=list(m.parents), names=list(m.names), fps=1.0 / m.frametime)
    side = Path(path).with_suffix('.npz')
    if Path(path).name.endswith('.footlock.bvh'):
        side = Path(str(path)[:-len('.footlock.bvh')] + '.npz')
    if side.exists():
        z = np.load(side)
        if 'cmd_xz_m' in z.files:
            clip['cmd'] = np.asarray(z['cmd_xz_m'], np.float64)
        if 'contact' in z.files:
            clip['contact'] = np.asarray(z['contact']) > 0.5
    return clip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bvh', nargs='+')
    ap.add_argument('-o', '--out', help='.mp4 (needs ffmpeg) or .gif; default: <first stem>.mp4 next to the first file')
    ap.add_argument('--seconds', type=float, default=None, help='only the first N seconds')
    ap.add_argument('--fps', type=float, default=30.0, help='output frame rate (frames are subsampled to it)')
    ap.add_argument('--overview', action='store_true', help='show the whole path instead of following the root')
    ap.add_argument('--size', type=float, default=4.0, help='panel size in inches')
    a = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import animation

    clips = [load(p) for p in a.bvh]
    n_frames = min(len(c['pos']) for c in clips)
    src_fps = clips[0]['fps']
    if a.seconds:
        n_frames = min(n_frames, int(a.seconds * src_fps))
    step = max(1, int(round(src_fps / a.fps)))
    frames = range(1, n_frames, step)                                     # frame 0 may be the rest pose

    fig = plt.figure(figsize=(a.size * len(clips), a.size))
    axes, artists = [], []
    for k, c in enumerate(clips):
        ax = fig.add_subplot(1, len(clips), k + 1, projection='3d')
        ax.set_title(c['name'], fontsize=8)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-60)
        ax.set_axis_off()
        p = c['pos'][:n_frames]
        lo, hi = p[..., [0, 2]].reshape(-1, 2).min(0) - 2, p[..., [0, 2]].reshape(-1, 2).max(0) + 2
        for g in np.arange(np.floor(lo[0]), hi[0], 0.5):                 # ground grid, 0.5 m
            ax.plot([g, g], [lo[1], hi[1]], 0, color='#dddddd', lw=0.5)
        for g in np.arange(np.floor(lo[1]), hi[1], 0.5):
            ax.plot([lo[0], hi[0]], [g, g], 0, color='#dddddd', lw=0.5)
        if 'cmd' in c:                                                    # world xz -> plot (x, z) on the ground
            ax.plot(c['cmd'][:n_frames, 0], c['cmd'][:n_frames, 1], 0, color='#6aa6d8', lw=3, alpha=0.5)
        ax.plot(p[:, 0, 0], p[:, 0, 2], 0, color='#e8913a', lw=1)
        bones = [ax.plot([], [], [], color='#333333', lw=2)[0] for j in range(len(c['parents'])) if c['parents'][j] >= 0]
        feet = [ax.plot([], [], [], 'o', ms=5, color='#999999')[0] for _ in FEET]
        axes.append(ax)
        artists.append((bones, feet))
        if a.overview:
            lo, hi = p[..., [0, 2]].reshape(-1, 2).min(0), p[..., [0, 2]].reshape(-1, 2).max(0)
            mid, half = (lo + hi) / 2, max((hi - lo).max() / 2, 1.0)
            ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half); ax.set_zlim(0, 2 * half)
    fig.tight_layout()

    def draw(t):
        for c, ax, (bones, feet) in zip(clips, axes, artists):
            p = c['pos'][t]
            b = 0
            for j, par in enumerate(c['parents']):
                if par < 0:
                    continue
                bones[b].set_data_3d([p[par, 0], p[j, 0]], [p[par, 2], p[j, 2]], [p[par, 1], p[j, 1]])
                b += 1
            for i, name in enumerate(FEET):
                j = c['names'].index(name)
                on = bool(c['contact'][t, i]) if 'contact' in c and t < len(c['contact']) else False
                feet[i].set_data_3d([p[j, 0]], [p[j, 2]], [p[j, 1]])
                feet[i].set_color('#d62728' if on else '#999999')
            if not a.overview:
                r = p[0]
                ax.set_xlim(r[0] - 1.2, r[0] + 1.2); ax.set_ylim(r[2] - 1.2, r[2] + 1.2); ax.set_zlim(0, 2.4)
        return []

    out = Path(a.out) if a.out else Path(a.bvh[0]).with_suffix('.mp4')
    anim = animation.FuncAnimation(fig, draw, frames=list(frames), interval=1000.0 / a.fps, blit=False)
    writer = animation.PillowWriter(fps=a.fps) if out.suffix == '.gif' else animation.FFMpegWriter(fps=a.fps, bitrate=4000)
    anim.save(str(out), writer=writer, dpi=100)
    print(f'{len(frames)} frames -> {out}')


if __name__ == '__main__':
    main()
