import torch
from torch import nn
from classification import Classifier

def relu(X: torch.Tensor) -> torch.Tensor:
    a = torch.zeros_like(X)
    return torch.max(a, X)


class MLPScratch(Classifier):
    """The MLP model implemented from scratch."""
    num_inputs: int
    lr: float

    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.normal(mean=0, std=sigma, size=(num_inputs, num_hiddens)))
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.normal(mean=0, std=sigma, size=(num_hiddens, num_outputs)))
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H = relu(torch.matmul(X, self.W1) + self.b1)
        return torch.matmul(H, self.W2) + self.b2

    # def configure_optimizers(self):
    #     return torch.optim.SGD(self.parameters(), lr=self.lr)

class MLP(Classifier):
    """The concrete MLP model."""
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)