"""The released model (Hugging Face model repo myshi/MotionPersona_V1), downloaded into checkpoints/ on first use."""
from pathlib import Path

HF_REPO = 'myshi/MotionPersona_V1'
FILES = ('prior.ckpt', 'codec.ckpt')
DEFAULT_DIR = Path(__file__).resolve().parents[1] / 'checkpoints'


def ensure_weights(ckpt_dir=DEFAULT_DIR, revision=None):
    """Download prior.ckpt + codec.ckpt into `ckpt_dir` unless both are there; returns the path of prior.ckpt."""
    ckpt_dir = Path(ckpt_dir)
    missing = [f for f in FILES if not (ckpt_dir / f).exists()]
    if missing:
        from huggingface_hub import hf_hub_download
        print(f'downloading {", ".join(missing)} from https://huggingface.co/{HF_REPO} -> {ckpt_dir}')
        for f in missing:
            hf_hub_download(HF_REPO, f, revision=revision, local_dir=ckpt_dir)
    return ckpt_dir / 'prior.ckpt'
