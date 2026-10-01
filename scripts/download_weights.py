"""Download the released model into checkpoints/ (generate.py and realtime_demo.py also do this on first use).

    python scripts/download_weights.py [--dir checkpoints]

Behind a firewall, set HF_ENDPOINT=https://hf-mirror.com (or download prior.ckpt / codec.ckpt by hand from
https://huggingface.co/myshi/MotionPersona_V1 into the same folder).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.weights import DEFAULT_DIR, ensure_weights  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default=str(DEFAULT_DIR))
    print(ensure_weights(ap.parse_args().dir))
