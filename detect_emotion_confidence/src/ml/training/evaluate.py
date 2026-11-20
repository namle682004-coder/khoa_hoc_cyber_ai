import torch
from torch.utils.data import DataLoader
from .metrics import accuracy


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
):
    """
    Evaluate model on validation/test dataset.
    """

    model.eval()

    total_loss = 0
    total_acc = 0
    total_samples = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_fn(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_acc += accuracy(outputs, labels) * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }