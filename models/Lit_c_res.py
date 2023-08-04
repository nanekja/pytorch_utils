import torch
import torch.nn as nn
from torchvision import models
from pytorch_lightning import LightningModule

from torch.nn import Linear, CrossEntropyLoss, functional as F
from torchmetrics.functional import accuracy

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, OneCycleLR, ReduceLROnPlateau
import torchmetrics
from torchmetrics import Accuracy


class Lit_c_resnet(LightningModule):

    def __init__(self, n_classes=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        #self.model = create_model()
        
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False), nn.MaxPool2d(2, 2), nn.BatchNorm2d(128), nn.ReLU())

        self.resblock1 = self.resblock(128, 128, 3)

        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False), nn.MaxPool2d(2, 2), nn.BatchNorm2d(256), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False), nn.MaxPool2d(2, 2), nn.BatchNorm2d(512), nn.ReLU())

        self.resblock2 = self.resblock(512, 512, 3)

        self.pool = nn.MaxPool2d(4,4)
        
        self.fc = nn.Linear(in_features=512,out_features=n_classes, bias = False)
        
        self.criterion = nn.CrossEntropyLoss()

        # optimizer parameters
        self.lr = lr

        self.accuracy = Accuracy(task='multiclass', num_classes=n_classes)

    def resblock(self, in_channels, out_channels, kernel_size):
        conv = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )
        return conv


    def forward(self, x):
        x = self.layer1(x) ## Input size = 32x32, output size = 32x32
        x = self.layer2(x) ## Input size = 32x32, output size = 16x16
        
        res_1 = self.resblock1(x) ## Input size = 16x16, output size = 16x16
        x = x + res_1
        
        x = self.layer3(x) ## Input size = 16x16, output size = 8x8
        x = self.layer4(x) ## Input size = 8x8, output size = 4x4
        
        res_2 = self.resblock2(x) ## Input size = 4x4, output size = 4x4
        x = x + res_2 
        
        x = self.pool(x) ## Input size = 4x4, output size = 1x1
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        x = x.view(-1, 10)
        
        return F.softmax(x, dim=-1)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)
        
    
    def configure_optimizers(self):
        '''defines model optimizer'''
        return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9,weight_decay=0.0001, nesterov=True)
        
