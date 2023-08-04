import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule



class CIFAR10DataModule(LightningDataModule):
    def __init__(self, batch_size=512, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.CIFAR10(root='./data', train=True, download=True)
        datasets.CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465), (0.247, 0.243, 0.261))
        ])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465), (0.247, 0.243, 0.261))
        ])

        if stage == 'fit' or stage is None:
            self.trainset = datasets.CIFAR10(root='./data', train=True, transform=transform_train)
            self.testset = datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)