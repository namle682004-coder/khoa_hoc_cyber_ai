import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import time
from pathlib import Path

from .config import load_config
from .data import get_dataloaders
from .model import SentimentRNN
from .utils import train_one_epoch, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, vocab = get_dataloaders(
        config, fast_dev_run=args.fast_dev_run
    )

    model = SentimentRNN(
        vocab_size=len(vocab.stoi),
        emb_dim=config["model"]["emb_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    total_start_time = time.time()

    for epoch in range(config["train"]["epochs"]):
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start_time

        print(
            f"Epoch {epoch+1}/{config['train']['epochs']} | "
            f"Time: {epoch_time:.2f}s | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            Path("outputs/checkpoints").mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "vocab_stoi": vocab.stoi,
                    "vocab_itos": vocab.itos,
                    "best_val_acc": best_acc,
                },
                "outputs/checkpoints/best.pt",
            )

            print("✅ Saved new best model")

    total_time = time.time() - total_start_time

    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    with open("outputs/reports/metrics.json", "w") as f:
        json.dump({"best_val_acc": best_acc}, f)

    print(f"\n🏁 Training finished")
    print(f"Best Val Acc = {best_acc:.4f}")
    print(f"Total Training Time = {total_time:.2f} seconds")


if __name__ == "__main__":
    main()