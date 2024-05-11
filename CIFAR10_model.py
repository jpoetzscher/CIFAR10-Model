import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

# for certificate issues
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

dtype = torch.float32
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
NUM_TRAIN = 49000
BATCH = 64

transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

cifar10_train = dset.CIFAR10('./DATA', train=True, transform=transform, download=True)
cifar10_val = dset.CIFAR10('./DATA', train=True, transform=transform, download=True)
cifar10_test = dset.CIFAR10('./DATA', train=False, transform=transform, download=True)

loader_train = DataLoader(cifar10_train, batch_size=BATCH, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(cifar10_val, batch_size=BATCH, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
loader_test = DataLoader(cifar10_test, batch_size=BATCH)


class Flatten(nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)

def eval(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, norm=True, active='GELU', pool=False, bias=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, bias=bias)]
        if norm==True:
            layers.append(nn.BatchNorm2d(out_channel))
        if active == 'GELU':
            layers.append(nn.GELU())
        elif active == 'RELU':
            layers.append(nn.ReLU())
        elif active == 'Leaky':
            layers.append(nn.LeakyReLU())
        if pool==True:
            layers.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv(x)
        return out

class FC(nn.Module):
    def __init__(self, hidden, num_classes, dropout):
        super().__init__()
        self.pool = nn.MaxPool2d(4)
        self.flatten = Flatten()
        self.out = nn.Linear(hidden, num_classes, bias=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(self.pool(x))
        out = self.dropout(self.out(x))
        return out
        


class CIFAR10_Model(nn.Module):
    def __init__(self, in_channel=3, channel_1=128, channel_2=256, channel_3=512, num_classes=10, active='GELU', norm=True, dropout=0.2, conv_bias=True):
        super().__init__()
        self.conv1 = Conv(in_channel, channel_1, norm=norm, active=active, pool=False, bias=conv_bias)
        self.conv2 = Conv(channel_1, channel_1, norm=norm, active=active, pool=True, bias=conv_bias)
        self.conv3 = Conv(channel_1, channel_2, norm=norm, active=active, pool=False, bias=conv_bias)
        self.conv4 = Conv(channel_2, channel_2, norm=norm, active=active, pool=True, bias=conv_bias)
        self.conv5 = Conv(channel_2, channel_3, norm=norm, active=active, pool=True, bias=conv_bias)
        self.conv6 = Conv(channel_3, channel_3, norm=norm, active=active, pool=False, bias=conv_bias)
        self.fc = FC(channel_3, num_classes, dropout=dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        out = self.fc(x)
        return out


model = CIFAR10_Model()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)

max_lr = 1e-2
optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-4)

print_every = 100

def train(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    def get_lr(optimizer):
      for param_group in optimizer.param_groups:
          return param_group['lr']
          
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=10, 
                                                steps_per_epoch=len(loader_train))
    
    #sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=766*3)
    model = model.to(device=device)
    for e in range(epochs):
        print("EPOCH: ", e)
        lrs = []
        print(len(loader_train))
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()

            loss.backward()

            # clip gradients
            nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()

            lrs.append(get_lr(optimizer))
            sched.step()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                eval(loader_val, model)
                print("Last learning rate: ", lrs[len(lrs)-1])
                print()

train(model, optimizer, epochs=10)
best_model = model
eval(loader_test, best_model)