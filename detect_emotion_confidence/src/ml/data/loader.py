from torch.utils.data import DataLoader

from .dataset import FER2013Dataset
from src.ml.preprocessing.image_transform import (
    get_train_transforms,
    get_test_transforms
)


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 64,
    num_workers: int = 4
):

    train_dataset = FER2013Dataset(
        root_dir=train_dir,
        transform=get_train_transforms()
    )

    test_dataset = FER2013Dataset(
        root_dir=test_dir,
        transform=get_test_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader