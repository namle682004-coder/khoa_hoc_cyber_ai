import os
import torch
from pathlib import Path


class Config:

    # =====================
    # DEVICE
    # =====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # PATHS
    # =====================

    BASE_DIR = Path(__file__).resolve().parents[2]

    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"

    RUNS_DIR = BASE_DIR / "runs"

    IMAGE_OUTPUT = RUNS_DIR / "images"
    VIDEO_OUTPUT = RUNS_DIR / "videos"
    WEBCAM_OUTPUT = RUNS_DIR / "webcam"

    os.makedirs(IMAGE_OUTPUT, exist_ok=True)
    os.makedirs(VIDEO_OUTPUT, exist_ok=True)
    os.makedirs(WEBCAM_OUTPUT, exist_ok=True)

    # =====================
    # DEFAULT PARAMS
    # =====================

    DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", 0.5))
    IMG_SIZE = int(os.getenv("IMG_SIZE", 640))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))

