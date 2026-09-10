from base import HyperParameters
import torch
import torchvision
from torchvision import transforms

class DataModule(HyperParameters):
    """The base class of data."""
    root: str = "../data"

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

class FashionMNIST(DataModule):
    """The Fashion-MNIST dataset."""
    batch_size: int

    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()

        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True) # downloads from the internet for training
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True) # downloads from the internet for validation

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                           num_workers=4)