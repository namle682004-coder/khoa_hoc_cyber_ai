import torch
import random
import numpy as np

from .train import train
from .loss import get_loss_fn
from .scheduler import get_scheduler
from .logger import Logger
from .early_stopping import EarlyStopping


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr=3e-4,
        epochs=20,
        weight_decay=1e-4,
        seed=42
    ):

        self.device = device
        self.epochs = epochs

        self._set_seed(seed)

        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # loss
        self.loss_fn = get_loss_fn()

        # scheduler
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_name="cosine",
            epochs=epochs
        )

        # logger
        self.logger = Logger()

        # early stopping
        self.early_stopping = EarlyStopping(
            patience=5
        )

        print("Trainer initialized")
        print(f"Epochs: {epochs}")
        print(f"LR: {lr}")
        print(f"Weight decay: {weight_decay}")

    def fit(self):

        train(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
            epochs=self.epochs,
            scheduler=self.scheduler,
            logger=self.logger,
            early_stopping=self.early_stopping
        )

    def _set_seed(self, seed):

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)