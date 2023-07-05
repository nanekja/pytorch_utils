import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import train1
import test1


def fit_model(net, device, train_loader, test_loader, optimizer, scheduler, NUM_EPOCHS=20):
    """Train+Test Model using train and test functions
    Args:
        net : torch model 
        NUM_EPOCHS : No. of Epochs
        device : "cpu" or "cuda" gpu 
        train_loader: Train set Dataloader with Augmentations
        test_loader: Test set Dataloader with Normalised images

    Returns:
        model, Tran Test Accuracy, Loss
    """
    training_acc, training_loss, testing_acc, testing_loss = list(), list(), list(), list()

    lr_trend = []
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1,NUM_EPOCHS+1):
        print("EPOCH: {} (LR: {})".format(epoch, optimizer.param_groups[0]['lr']))
        train_acc, train_loss, lr_hist = train1.train(net, device, train_loader, optimizer,scheduler)
        test_acc, test_loss = test1.test(net, device, test_loader)

        training_acc.append(train_acc)
        training_loss.append(train_loss)
        testing_acc.append(test_acc)
        testing_loss.append(test_loss)
        lr_trend.extend(lr_hist)
        
    return net, (training_acc, training_loss, testing_acc, testing_loss, lr_trend)