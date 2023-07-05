import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

#train_losses = []
#train_acc = []


def train(model, device, train_loader, optimizer,scheduler):
    """Model Training Loop
    Args:
        model : torch model 
        device : "cpu" or "cuda" gpu 
        train_loader : Torch Dataloader for trainingset
        optimizer : optimizer to be used
    Returns:
        float: accuracy and loss values
    """
    model.train()
    pbar = tqdm(train_loader)
    lr_trend = []
    correct = 0
    processed = 0
    num_loops = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
            # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch 
        # accumulates the gradients on subsequent backward passes. Because of this, when you start your training loop, 
        # ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)
        # Calculate loss
        loss = F.nll_loss(y_pred, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        scheduler.step()
        lr_trend.append(scheduler.get_last_lr()[0])

        train_loss += loss.item()
        
        # Update pbar-tqdm
        
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        num_loops += 1
        pbar.set_description(desc= f'Batch_id={batch_idx} Loss={train_loss/num_loops:.5f} Accuracy={100*correct/processed:0.2f}')

    return 100*correct/processed, train_loss/num_loops, lr_trend

