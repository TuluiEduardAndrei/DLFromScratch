import torch
from base import HyperParameters
from typing import Any

class Trainer(HyperParameters):
    """The base class for training models with data."""

    max_epochs: int
    num_train_batches: int
    num_val_batches: int
    train_dataloader: Any
    val_dataloader: Any
    optim: Any
    model: Any

    def __init__(self, max_epochs: int, num_gpus: int = 0, gradient_clip_val: float = 0):
        super().__init__()
        self.save_hyperparameters()

        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

    def prepare_data(self, data) -> None:
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model) -> None:
        model.trainer = self
        self.model = model

    def prepare_batch(self, batch):
        return batch

    def fit(self, model, data) -> None:
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()

        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self) -> None:
        self.model.train()

        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))

            self.optim.zero_grad()
            loss.backward()

            # if self.gradient_clip_val > 0:
            #     self.clip_gradients(self.gradient_clip_val, self.model)

            with torch.no_grad():
                self.optim.step()

            self.train_batch_idx += 1

        if self.val_dataloader is None:
            print(f"Epoch {self.epoch + 1}/{self.max_epochs} finished!")
            return

        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

        if self.model.val_losses:
            avg_loss = sum(self.model.val_losses) / len(self.model.val_losses)
            print(f"Validation Loss: {avg_loss:.4f}")
            self.model.val_losses.clear()

        print(f"Epoch {self.epoch + 1}/{self.max_epochs} finished!")