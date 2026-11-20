import torch.nn as nn


def get_loss_fn(name: str = "cross_entropy") -> nn.Module:
    """
    Factory function to create loss functions.
    """

    name = name.lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported loss function: {name}")