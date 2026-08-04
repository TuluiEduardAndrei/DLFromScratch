import inspect
from typing import Any
import torch

class HyperParameters:
    """The base class of hyperparameters."""

    def __init__(self):
        self.hparams: dict[str, Any] = {}

    def save_hyperparameters(self, ignore: list[Any] | None = None) -> None:
        """Save function arguments into class attributes."""
        if ignore is None:
            ignore = []

        # Extracting the current frame
        current_frame = inspect.currentframe()
        assert current_frame is not None, "The current frame could not be found."

        # Extracting the previous frame
        frame = current_frame.f_back
        assert frame is not None, "The previous frame could not be found."

        _, _, _, local_vars = inspect.getargvalues(frame)

        # Building the dictionary
        self.hparams = {}

        for k, v in local_vars.items():
            if k not in set(ignore+['self']) and not k.startswith('_'):
                self.hparams[k] = v

        # Setting the attributes
        for k, v in self.hparams.items():
            setattr(self, k, v)


class Module(torch.nn.Module, HyperParameters):
    """The base (skelet) class of models."""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.val_losses: list = []

    # To be implemented by the concrete model.
    def loss(self, y_hat, y):
        raise NotImplementedError

    # To be implemented by the concrete model.
    def forward(self, X):
        raise NotImplementedError

    def training_step(self, batch) -> torch.Tensor:
        l = self.loss(self(*batch[:-1]), batch[-1])
        return l

    def validation_step(self, batch) -> None:
        l: torch.Tensor = self.loss(self(*batch[:-1]), batch[-1])
        self.val_losses.append(l.item())

    # To be implemented by the concrete model.
    def configure_optimizers(self):
        raise NotImplementedError

class DataModule(HyperParameters):
    """The base class of data."""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    # To be implemented by the concrete data module
    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)