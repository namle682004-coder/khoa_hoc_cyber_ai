import torch
from itertools import product

from .train import train
from .loss import get_loss_fn


def grid_search(
    model_fn,
    train_loader,
    val_loader,
    device,
):

    learning_rates = [1e-3, 1e-4]
    batch_sizes = [32, 64]

    best_acc = 0
    best_config = None

    for lr, bs in product(learning_rates, batch_sizes):

        print(f"\nTesting lr={lr}, batch_size={bs}")

        model = model_fn()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        loss_fn = get_loss_fn()

        train(
            model,
            train_loader,
            val_loader,
            optimizer,
            loss_fn,
            device,
            epochs=5
        )

        # evaluation logic có thể thêm ở đây

        # placeholder
        acc = 0

        if acc > best_acc:

            best_acc = acc
            best_config = (lr, bs)

    print("Best config:", best_config)