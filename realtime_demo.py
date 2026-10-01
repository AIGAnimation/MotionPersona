"""Interactive realtime demo: steer a character with the keyboard / a gamepad while switching persona, style and body.

    python realtime_demo.py                                   # released model, p02 / neutral on the average body
    python realtime_demo.py --subject p10 --style happy
    python realtime_demo.py --headless --seed 0 --script scripts/demo_walk.json --export outputs/stream
    python realtime_demo.py --help                            # every option

Needs the SMPL-X neutral model (not redistributed): download SMPLX_NEUTRAL.npz from https://smpl-x.is.tue.mpg.de and
point SMPLX_NPZ at it (or place it at data/body_models/smplx/SMPLX_NEUTRAL.npz). The skinned body mesh
(realtime/assets/smplx_neutral.glb) is built from it on the first run.

Controls: WASD / left stick move, Shift / L3 run, F lock the facing (then A/D strafe, S backpedal), Q/E turn the
facing, right-mouse drag sets it; [ ] style, , . persona, B morph to the persona's own body, 10 beta sliders on the
right morph the body, M model on/off, R restart the stream, L foot lock, U panels, H HUDs.
"""
import os
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RT = HERE / 'realtime'
sys.path.insert(0, str(RT))
sys.path.insert(0, str(HERE))

from definitions_smplx import SMPLX_NPZ  # noqa: E402
from utils.weights import ensure_weights  # noqa: E402

if not Path(SMPLX_NPZ).exists():
    raise SystemExit('SMPL-X model not found: download SMPLX_NEUTRAL.npz from https://smpl-x.is.tue.mpg.de and set '
                     'SMPLX_NPZ=/path/to/SMPLX_NEUTRAL.npz (see README.md)')
if not (RT / 'assets/smplx_neutral.glb').exists():
    print('building the SMPL-X body mesh (once) ...')
    subprocess.run([sys.executable, str(RT / 'smplx_glb.py'), '--texture', os.environ.get('SMPLX_TEXTURE', 'none')],
                   check=True, cwd=RT)

if '--model' not in sys.argv:
    ensure_weights()

# repo-relative paths in the demo (e.g. --script / --export) are taken from the caller's working directory as usual
sys.argv[0] = str(RT / 'Program.py')
runpy.run_path(str(RT / 'Program.py'), run_name='__main__')
