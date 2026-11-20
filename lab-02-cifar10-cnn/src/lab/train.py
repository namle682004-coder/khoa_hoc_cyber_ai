import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from .model import SimpleCNN
from .data import get_dataloaders
from .utils import train_one_epoch, evaluate
from .config import load_config


def main(config_path, fast_dev_run=False):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(config, fast_dev_run)

    model = SimpleCNN().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(config.epochs if not fast_dev_run else 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        scheduler.step()

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch {epoch+1}: "
            f"Train Acc={train_acc:.4f} "
            f"Val Acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "outputs/checkpoints/best.pt")

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Acc={test_acc:.4f}")

    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    with open("outputs/reports/metrics.json", "w") as f:
        json.dump({
            "best_val_acc": best_acc,
            "test_acc": test_acc
        }, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast_dev_run", action="store_true")

    args = parser.parse_args()

    main(args.config, args.fast_dev_run)