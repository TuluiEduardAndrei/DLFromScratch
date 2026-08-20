from base import Module, HyperParameters
import torch
from torch import nn
from torch.nn import functional as F
from classification import Classifier

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

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

class SoftmaxRegressionScratch(Classifier):
    """The softmax regression model implemented from scratch."""
    lr: float

    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return softmax(torch.matmul(X, self.W) + self.b)

    def configure_optimizers(self):
        return SGD(self.parameters(), self.lr)

    def parameters(self):
        return [self.W, self.b]

    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)

class SoftmaxRegression(Classifier):
    """The concrete softmax regression model."""
    lr: float

    def __init__(self, num_outputs, lr):
         super().__init__()
         self.save_hyperparameters()

         self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)

    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), self.lr)

    def loss(self, y_hat, y, averaged=True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))     # Y_hat: torch.Size([256, 10])
        y = y.reshape((-1,))     # Y: torch.Size([256])
        return F.cross_entropy(y_hat, y, reduction='mean' if averaged else 'none')