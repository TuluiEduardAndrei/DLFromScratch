from base import Module
import torch

class Classifier(Module):
    """The base class of classification models."""
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.val_accuracy: list = []

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        Y_true = batch[-1]

        l = self.loss(Y_hat, Y_true)
        a = self.accuracy(Y_hat, Y_true)

        self.val_losses.append(l.item())
        self.val_accuracy.append(a.item())

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        predictions = Y_hat.argmax(axis=1).type(Y.dtype)
        compare = (predictions == Y.reshape(-1)).type(torch.float32)

        return compare.mean() if averaged else compare