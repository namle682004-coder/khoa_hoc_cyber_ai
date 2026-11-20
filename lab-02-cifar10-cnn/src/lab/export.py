import torch
import json
from pathlib import Path
from .model import SimpleCNN
from .config import load_config


CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def export_torchscript(model):
    model.eval()
    example = torch.randn(1, 3, 32, 32)
    traced = torch.jit.trace(model, example)

    Path("outputs/export").mkdir(parents=True, exist_ok=True)
    traced.save("outputs/export/model.ts")

    with open("outputs/export/classes.json", "w") as f:
        json.dump(CLASSES, f)


def main(config_path):
    config = load_config(config_path)

    model = SimpleCNN()
    model.load_state_dict(
        torch.load("outputs/checkpoints/best.pt", map_location="cpu")
    )

    export_torchscript(model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True)
    args = parser.parse_args()

    main(args.config)