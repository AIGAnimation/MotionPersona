"""On-screen HUDs for recording the demo.

  InputHUD  (bottom-left): the stick / WASD state, run and facing-lock, and the three control axes -- persona, style,
            body height -- in the paper's axis colours.
  TimingHUD (top-right):   the last chunk's compute time against the 3-frame budget (100 ms at 30 fps), and the
            device it ran on.

Both are drawn in window pixels scaled from a 1080-line layout, so text is >= 28 px at 1080p. Nothing is drawn into
the bottom 12 % of the frame (subtitle area). Program places them clear of its panels when the UI is shown.
"""
import numpy as np
import raylib as rl

from ai4animation import AI4Animation

# the paper's colours for the three control axes
BODY = (42, 120, 214, 255)        # #2a78d6
PERSONA = (235, 104, 52, 255)     # #eb6834
STYLE = (138, 95, 201, 255)       # #8a5fc9
INK = (61, 61, 59, 255)           # #3d3d3b
MUTED = (155, 154, 149, 255)      # #9b9a95
PANEL = (255, 255, 255, 205)
CAP_OFF = (236, 235, 232, 255)
TEXT_PX = 30                      # >= 28 px at 1080p
SMALL_PX = 28
BUDGET_MS = 100.0                 # 3 frames at 30 fps: a chunk must be back before the committed frames run out
SUBTITLE_TOP = 0.88               # keep out of the bottom 12 %


def _scale():
    return AI4Animation.Draw.ScreenHeight() / 1080.0


def _text(s, x, y, px, color, pivot=0):
    """Text at pixel (x, y) (top-left, or top-right with pivot=1), px tall."""
    H, W = AI4Animation.Draw.ScreenHeight(), AI4Animation.Draw.ScreenWidth()
    AI4Animation.Draw.Text(s, x / W, y / H, px / H, color, pivot=pivot)


def _panel(x, y, w, h, s):
    rl.DrawRectangleRounded((float(x), float(y), float(w), float(h)), 0.12, 8, PANEL)
    rl.DrawRectangleRoundedLinesEx((float(x), float(y), float(w), float(h)), 0.12, 8, max(1.0, 1.5 * s),
                                   (215, 214, 210, 255))


def _cap(label, x, y, w, h, on, s, color=INK):
    rl.DrawRectangleRounded((float(x), float(y), float(w), float(h)), 0.25, 6, color if on else CAP_OFF)
    _text(label, x + w / 2, y + (h - 24 * s) / 2, 24 * s, (255, 255, 255, 255) if on else MUTED, pivot=0.5)


def device_label(device, threads=None, name=None):
    """'RTX 5080 (CUDA)' / 'CPU, 4 threads' / 'MPS'. ASCII only: the demo's TTF is loaded with raylib's default
    (ASCII) glyph set, so a middle dot renders as '?'."""
    if device == 'cuda':
        n = (name or 'GPU').replace('NVIDIA ', '').replace('GeForce ', '')
        return f'{n} (CUDA)'
    if device == 'cpu':
        return f'CPU, {threads} threads' if threads else 'CPU'
    return str(device).upper()


def draw_input_hud(x0, stick, run, facing_locked, subject, style, height_m, body_label=None):
    """Bottom-left block. `x0` = left edge as a fraction of the window width; `stick` = (x, y) in the WASD
    convention (+y = W); `height_m` = the current body's mesh height."""
    s = _scale()
    W, H = AI4Animation.Draw.ScreenWidth(), AI4Animation.Draw.ScreenHeight()
    w, h = 470 * s, 290 * s
    x, y = x0 * W, SUBTITLE_TOP * H - h - 12 * s
    _panel(x, y, w, h, s)

    # stick: ring + knob (+ a run halo)
    r = 50 * s
    cx, cy = x + 22 * s + r, y + 20 * s + r
    rl.DrawCircle(int(cx), int(cy), r, CAP_OFF)
    rl.DrawCircleLines(int(cx), int(cy), r, MUTED)
    sx, sy = float(np.clip(stick[0], -1, 1)), float(np.clip(stick[1], -1, 1))
    n = np.hypot(sx, sy)
    if n > 1.0:
        sx, sy = sx / n, sy / n
    kx, ky = cx + sx * r * 0.72, cy - sy * r * 0.72
    if n > 1e-3:
        rl.DrawLineEx((float(cx), float(cy)), (float(kx), float(ky)), 4 * s, MUTED)
    if run:
        rl.DrawCircle(int(kx), int(ky), 21 * s, (61, 61, 59, 70))
    rl.DrawCircle(int(kx), int(ky), 14 * s, INK)

    # WASD caps + Shift (run) + F (facing lock)
    c, g = 38 * s, 6 * s
    kx0, ky0 = cx + r + 26 * s, y + 14 * s
    keys = {'W': sy > 0.3, 'A': sx < -0.3, 'S': sy < -0.3, 'D': sx > 0.3}
    _cap('W', kx0 + c + g, ky0, c, c, keys['W'], s)
    for i, k in enumerate('ASD'):
        _cap(k, kx0 + i * (c + g), ky0 + c + g, c, c, keys[k], s)
    _cap('Shift', kx0, ky0 + 2 * (c + g), 2 * c + g, c, run, s)
    _cap('F', kx0 + 2 * (c + g), ky0 + 2 * (c + g), c, c, facing_locked, s)
    tx = kx0 + 3 * (c + g) + 10 * s
    _text('run' if run else 'walk', tx, ky0 + 4 * s, SMALL_PX * s, INK if run else MUTED)
    _text('face lock' if facing_locked else 'face move', tx, ky0 + 2 * (c + g) + 4 * s, 22 * s, MUTED)

    # the three axes
    ly = ky0 + 3 * (c + g) + 14 * s                     # below the caps
    lx, vx = x + 22 * s, x + 185 * s
    rows = (('persona', PERSONA, str(subject)), ('style', STYLE, str(style)),
            ('body', BODY, f'{height_m * 100:.0f} cm' + (f'  {body_label}' if body_label else '')))
    for i, (name, color, value) in enumerate(rows):
        yy = ly + i * 36 * s
        rl.DrawCircle(int(lx + 7 * s), int(yy + TEXT_PX * s / 2), 7 * s, color)
        _text(name, lx + 22 * s, yy, TEXT_PX * s, color)
        _text(value, vx, yy, TEXT_PX * s, INK)


def draw_timing_hud(x1, ms, ms_avg, device, hop=None):
    """Top-right block. `x1` = right edge as a fraction of the window width; `ms` = the last chunk's compute time."""
    s = _scale()
    W, H = AI4Animation.Draw.ScreenWidth(), AI4Animation.Draw.ScreenHeight()
    w, h = 420 * s, 156 * s
    x, y = x1 * W - w, 0.02 * H
    _panel(x, y, w, h, s)
    px = x + 20 * s
    _text('block', px, y + 14 * s, TEXT_PX * s, MUTED)
    _text(f'{ms:.0f} ms', px + 90 * s, y + 14 * s, TEXT_PX * s, INK)
    if hop:
        _text(f'hop {hop}', x + w - 20 * s, y + 14 * s, SMALL_PX * s, MUTED, pivot=1)
    # bar: the 100 ms budget, filled by this chunk
    bx, by, bw, bh = px, y + 56 * s, w - 40 * s, 16 * s
    rl.DrawRectangleRounded((float(bx), float(by), float(bw), float(bh)), 0.5, 6, CAP_OFF)
    f = float(np.clip(ms / BUDGET_MS, 0.0, 1.0))
    if f > 0:
        rl.DrawRectangleRounded((float(bx), float(by), float(max(bw * f, bh)), float(bh)), 0.5, 6,
                                INK if ms <= BUDGET_MS else PERSONA)
    _text(f'budget 100 ms (3 frames)  {100 * ms / BUDGET_MS:.0f} %', px, y + 80 * s, 22 * s, MUTED)
    _text(device, px, y + 110 * s, SMALL_PX * s, INK)
    if ms_avg:
        _text(f'med {ms_avg:.0f}', x + w - 20 * s, y + 110 * s, SMALL_PX * s, MUTED, pivot=1)
