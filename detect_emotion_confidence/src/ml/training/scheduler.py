import torch


def get_scheduler(optimizer, scheduler_name="cosine", epochs=20):

    if scheduler_name == "cosine":

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

        return scheduler

    if scheduler_name == "step":


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

        return scheduler

    return None