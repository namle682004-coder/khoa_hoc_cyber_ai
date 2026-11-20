import argparse
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from ultralytics import YOLO

from config import Config
from utils import download_dataset, calculate_map


# ======================================
# Training Function
# ======================================

def train_yolo(args, config):
    """
    1. Load data từ Roboflow hoặc local
    2. Initialize YOLOv8 model
    3. Training loop với validation
    4. Save best checkpoint based on mAP50-95
    5. Plot training curves
    6. Early stopping
    """

    # -----------------------
    # Dataset
    # -----------------------
    if args.download:
        print("[INFO] Downloading data from Roboflow...")
        dataset_path = download_dataset(
            dataset_name=config.DATASET_NAME,
            api_key=args.roboflow_key,
            workspace=args.workspace,
            version=args.version,
            target_dir=config.DATA_ROOT,
        )
        data_yaml = os.path.join(dataset_path, "data.yaml")
    else:
        data_yaml = config.DATA_YAML

    print(f"[INFO] Using data: {data_yaml}")

    # -----------------------
    # Model init
    # -----------------------
    model_name = f"yolov8{args.model_size}.pt"
    model = YOLO(model_name)

    # -----------------------
    # Training
    # -----------------------
    print("[INFO] Start training...")

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=config.LEARNING_RATE,
        device=config.DEVICE,
        project="runs",
        name="train.sh",
        patience=config.EARLY_STOPPING_PATIENCE if config.EARLY_STOPPING else 0,
    )

    # -----------------------
    # Save best checkpoint
    # -----------------------
    best_weight = Path(model.trainer.save_dir) / "weights" / "best.pt"
    target_path = Path(config.CHECKPOINT_DIR) / "best.pt"

    shutil.copy(best_weight, target_path)

    print(f"[INFO] Best checkpoint saved to: {target_path}")

    # -----------------------
    # Final validation metrics
    # -----------------------
    metrics = calculate_map(model, data_yaml)

    print("\nFinal Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # -----------------------
    # Plot curves
    # -----------------------
    plot_training_curves(model.trainer.save_dir)


# ======================================
# Plot training curves
# ======================================

def plot_training_curves(save_dir):

    csv_path = Path(save_dir) / "results.csv"

    if not csv_path.exists():
        print("[WARN] results.csv not found, skip plotting.")
        return

    import pandas as pd

    df = pd.read_csv(csv_path)

    fig_dir = Path(save_dir) / "plots"
    fig_dir.mkdir(exist_ok=True)

    # Loss
    plt.figure()
    plt.plot(df["train.sh/box_loss"], label="box_loss")
    plt.plot(df["train.sh/cls_loss"], label="cls_loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig(fig_dir / "loss.png")

    # mAP
    plt.figure()
    plt.plot(df["metrics/mAP50(B)"], label="mAP50")
    plt.plot(df["metrics/mAP50-95(B)"], label="mAP50-95")
    plt.legend()
    plt.title("mAP")
    plt.savefig(fig_dir / "map.png")

    # Precision / Recall
    plt.figure()
    plt.plot(df["metrics/precision(B)"], label="Precision")
    plt.plot(df["metrics/recall(B)"], label="Recall")
    plt.legend()
    plt.title("Precision & Recall")
    plt.savefig(fig_dir / "pr.png")

    print(f"[INFO] Training curves saved to {fig_dir}")


# ======================================
# CLI
# ======================================

def parse_args():
    parser = argparse.ArgumentParser("YOLOv8 Training")

    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=Config.IMAGE_SIZE)
    parser.add_argument("--model-size", type=str, default=Config.MODEL_SIZE)

    # Roboflow optional
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--roboflow-key", type=str, default=None)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--version", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train_yolo(args, Config)
