
import torch
import dataloader
import albumentation

SEED = 69

torch.manual_seed(SEED)

batch_size=32

trainset = dataloader.Cifar10Dataset(root='./data', train=True, download=True, transform=albumentation.train_transforms())
testset = dataloader.Cifar10Dataset(root='./data', train=False, download=True, transform=albumentation.test_transforms())

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
