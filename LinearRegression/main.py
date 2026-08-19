from linear_regression import LinearRegressionScratch
from data import SyntheticRegressionData
from trainer import Trainer
import torch
from torchviz import make_dot
from linear_regression import LinearRegression


def main():
    print("\nFrom Scratch\n")
    model = LinearRegressionScratch(3, lr=0.05)
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4, 7]), b=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)

    print("\n")
    with torch.no_grad():
        print(f'Error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
        print(f'Error in estimating b: {data.b - model.b}')

    if model.w.grad is not None:
        print(f'model.w.grad: {model.w.grad}')
    else:
        print('No gradient found for w')
    if model.b.grad is not None:
        print(f'model.b.grad: {model.b.grad}')
    else:
        print('No gradient found for b')

    X_dummy = torch.randn(1, 3)
    y_hat = model(X_dummy)
    visual_graph = make_dot(y_hat, params={'w': model.w, 'b': model.b})
    visual_graph.render()

    """
        The estimation error is close to 0, meaning that the model managed to deduce the hidden w and b laws
    almost perfectly. 
        The success is also shown by the value of the gradient, which is very close to 0,
    and when a gradient is 0, that means the model found the minimum point for the loss function.
    """

    print("\n----------------------------------------------------------------------------------------\n")

    print("Using PyTorch API\n")

    model = LinearRegression(lr=0.05)
    data = SyntheticRegressionData(w=torch.tensor([2, -3.4, 7]), b=4.2)
    trainer = Trainer(max_epochs=3)
    trainer.fit(model, data)

    print("\n")
    w, b = model.parameters()

    with torch.no_grad():
        print(f'Error in estimating w: {data.w - w.reshape(data.w.shape)}')
        print(f'Error in estimating b: {data.b - b}')

    if w.grad is not None:
        print(f'w.grad: {w.grad}')
    else:
        print('No gradient found for w')
    if b.grad is not None:
        print(f'b.grad: {b.grad}')
    else:
        print('No gradient found for b')


if __name__ == '__main__':
    main()