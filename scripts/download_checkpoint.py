from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "saved" / "soundstream-final"
REPO_ID = "9imon4ik/soundstream-neural-audio-codec"
FILENAME = "checkpoint-epoch100.pth"

CKPT_DIR.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=str(CKPT_DIR),
)
print(f"Checkpoint: {path}")
