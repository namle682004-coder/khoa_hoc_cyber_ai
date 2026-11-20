
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

import os
class Config:
    # ========================
    # Device & Reproducibility
    # ========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    # ========================
    # Dataset settings
    # ========================
    DATASET_NAME = "hard-hat-detection"
    DATA_ROOT = "./data"

    NUM_CLASSES = 2
    CLASS_NAMES = ["helmet", "no-helmet"]

    # ========================
    # Training hyperparameters
    # ========================
    EPOCHS = 50
    BATCH_SIZE = 8
    IMAGE_SIZE = 512
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.937

    NUM_WORKERS = 4
    PIN_MEMORY = True

    # ========================
    # YOLO model settings
    # ========================
    MODEL_SIZE = "n"      # n / s / m / l / x
    PRETRAINED = True

    YOLO_VERSION = "v8"
    MODEL_NAME = f"yolov8{MODEL_SIZE}.pt"

    # ========================
    # Checkpoints
    # ========================
    CHECKPOINT_DIR = "./checkpoints"
    SAVE_BEST_ONLY = True
    SAVE_INTERVAL = 5   # save mỗi N epoch

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ========================
    # Early stopping
    # ========================
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10

    # ========================
    # Logging
    # ========================
    LOG_INTERVAL = 100

    # ========================
    # Data augmentation
    # ========================
    USE_AUGMENTATION = True
    AUGMENTATION_PARAMS = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
    }

    # ========================
    # Inference
    # ========================
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    TOP_K = 5

    # ========================
    # Paths
    # ========================
    TRAIN_IMAGES = os.path.join(DATA_ROOT, "train.sh/images")
    VAL_IMAGES = os.path.join(DATA_ROOT, "valid/images")
    TEST_IMAGES = os.path.join(DATA_ROOT, "test/images")

    DATA_YAML = os.path.join(DATA_ROOT, "data.yaml")

def print_config(cls):
    print("=" * 50)
    print("YOLO Object Detection Configuration")
    print("=" * 50)

    print(f"Device: {cls.DEVICE}")
    print(f"Seed: {cls.SEED}")

    print(f"Dataset: {cls.DATASET_NAME}")
    print(f"Classes: {cls.NUM_CLASSES}")
    print(f"Class names: {cls.CLASS_NAMES}")

    print(f"Model: {cls.MODEL_NAME}")
    print(f"Pretrained: {cls.PRETRAINED}")

    print(f"Batch size: {cls.BATCH_SIZE}")
    print(f"Learning rate: {cls.LEARNING_RATE}")
    print(f"Epochs: {cls.EPOCHS}")

    print(f"Early stopping: {cls.EARLY_STOPPING}")
    print(f"Early stopping patience: {cls.EARLY_STOPPING_PATIENCE}")

    print(f"Augmentation: {cls.USE_AUGMENTATION}")

    print("=" * 50)
