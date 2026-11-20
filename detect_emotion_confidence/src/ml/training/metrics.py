import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    """

    pred_labels = torch.argmax(preds, dim=1)

    correct = (pred_labels == targets).sum().item()

    total = targets.size(0)

    return correct / total