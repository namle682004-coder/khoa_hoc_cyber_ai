import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),

        transforms.RandomErasing(p=0.25)
    ])
    val_tf = transforms.Compose([
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
        transforms.ToTensor(),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    return train_tf, val_tf


def get_dataloaders(config, fast_dev_run=False):

    train_tf, val_tf = get_transforms()

    full_train_ds = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_tf
    )

    test_ds = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=val_tf
    )

    # ===== train / validation split =====
    train_size = int(0.9 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size

    generator = torch.Generator().manual_seed(42)

    train_ds, val_ds = random_split(
        full_train_ds,
        [train_size, val_size],
        generator=generator
    )

    if fast_dev_run:
        train_ds = Subset(train_ds, range(256))
        val_ds = Subset(val_ds, range(256))

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader