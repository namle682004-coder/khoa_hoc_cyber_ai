import torch
import random
import numpy as np

from src.ml.data.loader import create_dataloaders
from src.ml.models.model import EmotionResNet18
from src.ml.training import Trainer


def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", torch.cuda.get_device_properties(0).total_memory // 1e9, "GB")

    # dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dir="data/raw/fer2013/train",
        test_dir="data/raw/fer2013/test",
        batch_size=64
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    # model
    model = EmotionResNet18(
        pretrained=True
    )

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=30,      # tăng epoch
        lr=3e-4
    )

    trainer.fit()


if __name__ == "__main__":
    main()