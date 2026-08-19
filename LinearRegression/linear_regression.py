from base import Module, HyperParameters
import torch
from torch import nn

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


class LinearRegression(Module):
    """The linear regression model implemented with high-level APIs."""

    lr: float

    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X) -> torch.Tensor:
        return self.net(X)

    def loss(self, y_hat, y) -> torch.Tensor:
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self) -> SGD:
        return torch.optim.SGD(self.parameters(), self.lr)

    def parameters(self):
        return [self.net.weight, self.net.bias]