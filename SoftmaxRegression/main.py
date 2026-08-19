from softmax_regression import SoftmaxRegressionScratch
from data import FashionMNIST
from trainer import Trainer
import torch


def main():
    print("\nFrom Scratch\n")
    data = FashionMNIST(batch_size=256)
    model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.2)
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, data)

    labels = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]

    X, y = next(iter(data.val_dataloader()))

    with torch.no_grad():
        y_hat = model(X)

    predictions = y_hat.argmax(dim=1)

    for i in range(10):
        print(f"True: {labels[y[i]]:<12}\tPredicted: {labels[predictions[i]]}")

if __name__ == '__main__':
    main()