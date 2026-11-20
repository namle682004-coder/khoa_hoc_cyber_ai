import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluate import evaluate
from .checkpoint import save_checkpoint


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    loss_fn,
    device,
    epochs: int = 10,
    scheduler=None,
    logger=None,
    early_stopping=None
):

    model.to(device)

    best_acc = 0.0

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        total_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in progress:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward
            with torch.amp.autocast("cuda"):

                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            progress.set_postfix(
                loss=running_loss / total_samples
            )

        train_loss = running_loss / total_samples

        # Validation
        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device
        )

        val_acc = val_metrics["accuracy"]
        val_loss = val_metrics["loss"]

        # Save best model
        if val_acc > best_acc:

            best_acc = val_acc

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_acc,
                path="checkpoints/best_model.pt"
            )

            print(f"🔥 New best model saved | Acc: {best_acc:.4f}")

        # Scheduler step
        if scheduler is not None:

            try:
                scheduler.step(val_acc)
            except TypeError:
                scheduler.step()

        # Logger
        if logger is not None:

            logger.log_train(train_loss, epoch)

            logger.log_val(
                val_loss,
                val_acc,
                epoch
            )

        # Early stopping
        if early_stopping is not None:

            stop = early_stopping.step(val_acc)

            if stop:
                print("Early stopping triggered")
                break

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Best Acc: {best_acc:.4f}"
        )