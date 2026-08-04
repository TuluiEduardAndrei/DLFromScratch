from base import Module, HyperParameters, DataModule
import torch

class SGD(HyperParameters):
    """Minibatch stochastic gradient descent."""

    def __init__(self, params, lr):
        super().__init__()
        self.params = params
        self.lr = lr

        self.save_hyperparameters()

    def step(self) -> None:
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class LinearRegressionScratch(Module):
    """The linear regression model implemented from scratch."""

    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X) -> torch.Tensor:
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y) -> torch.Tensor:
        l = (y_hat - y) ** 2 / 2
        return l.mean()

    def configure_optimizers(self) -> SGD:
        return SGD([self.w, self.b], self.lr)

class SyntheticRegressionData(DataModule):
    """Synthetic data for linear regression."""

    num_train: int
    batch_size: int

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape(-1, 1)) + b + noise

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)