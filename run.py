#!python3

import torch
import torch.nn as nn
import os, sys
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torchinfo import summary
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK']='True'

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, 'models'))
from dvh import dvhNet

def train_step(dataloader, model, loss_fn, optimizer, device='cpu'):
    '''train operations for one epoch'''
    size = len(dataloader.dataset) # number of samples
    for batch_idx, (images, y) in enumerate(dataloader):
        images, y = images.to(device), y.to(device) # TODO
        print(images.shape)
        points = torch.zeros(1, 3, 4) # TODO
        pred = model(images, points) # predicts on the batch of training data
        print(y.shape)
        print(pred.shape)
        loss = loss_fn(pred, y) # compute prediction error

        # Backpropagation of predication error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
	
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(images) # (batch size)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    training_data = datasets.FakeData(transform=ToTensor())
    # test_data = datasets.FashionMNIST(root="data", train=False, download=True)
    # X: torch.Size([64, 1, 28, 28]); y: torch.Size([64])
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    model = dvhNet()
    # summary(model, [(1,3,224,224), (1, 3, 4)])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch_idx in range(epochs):
        print(f"Epoch {epoch_idx+1}\n-------------------------------")
        train_step(train_dataloader, model, loss_fn, optimizer)
    print("Done!")

