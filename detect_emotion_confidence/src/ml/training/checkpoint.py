import torch
from pathlib import Path


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_acc: float,
    path: str,
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path: str):

    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint["epoch"], checkpoint["val_acc"]