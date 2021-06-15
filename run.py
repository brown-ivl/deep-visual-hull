#!python3


import os, sys
import argparse
import time

import torch
import torch.nn as nn
from torchinfo import summary
import torchvision
from torch.utils.tensorboard import SummaryWriter
print(torch.__version__)
print(torchvision.__version__)

import numpy as np

import config
from data import CustomImageDataset
import util

writer = SummaryWriter()
flags = None
os.environ['KMP_DUPLICATE_LIB_OK']='True'

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, 'models'))
from dvhNet import dvhNet


def train_step(dataloader, model, loss_fn, optimizer, device='cpu'):
    '''train operations for one epoch'''
    # size = len(dataloader.dataset) # number of samples
    for batch_idx, (images, points, y) in enumerate(dataloader):
        images, points, y = images.to(device), points.to(device), y.to(device) 
        pred = model(images.float(), points.float()) # predicts on the batch of training data
        reshaped_pred = pred.transpose(1, 2) # (batch_size, T=8, 1)
        reshaped_pred = reshaped_pred.reshape((config.batch_size, config.resolution, config.resolution, config.resolution))
        loss = loss_fn(reshaped_pred.float(), y.float()) # compute prediction error
        # Backpropagation of predication error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
	
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(images) # (batch size)
            print(f"loss: {loss:>7f}") # [{current:>5d}/{size:>5d}]

    return loss

def test(dataloader, model, loss_fn, threshold=0.3, device='cpu'):
    '''model: with loaded checkpoint or trained parameters'''
    for batch_idx, (images, points, y) in enumerate(dataloader):
        print("----", batch_idx)
        images, points, y = images.to(device), points.to(device), y.to(device)  # points: (batch_size, 3, T)
        pred = model(images.float(), points.float()) # (batch_size, 1, T=16**3=4096)
        print("pred\n", pred)
        # loss = loss_fn(reshaped_pred.float(), y.float())
        ## convert prediction to point cloud, then to voxel grid
        indices = torch.nonzero(pred>threshold, as_tuple=True)
        print("indices\n", indices)
        pointcloud = np.array(points[indices[0], :, indices[2]]) # array of [x,y,z] where pred > threshold
        print("pointcloud\n", pointcloud)
        # TODO: check pointcloud has element (none of the dimension is 0)
        voxel = util.pointcloud2voxel(pointcloud, config.resolution)
        util.draw_voxel_grid(voxel)


if __name__ == "__main__":
    timestamp = str(int(time.time()))
    print(f"################# Timestamp = {timestamp} #################")

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_ckpt_dir', type=str, default=config.save_ckpt_dir, help="The directory to store the .pth model files")
    parser.add_argument('--load_ckpt_dir', type=str, help="The directory to load .pth model files from, required for test mode")
    parser.add_argument('--mode', type=str, default="train", help="One of 'train' or 'test'")
    flags, unparsed = parser.parse_known_args()
    
    if flags.mode=="train":
        flags.save_ckpt_dir += f'{timestamp}/' if flags.save_ckpt_dir.endswith("/") else f'{timestamp}/'
        if os.path.exists(flags.save_ckpt_dir) == False:
            os.makedirs(flags.save_ckpt_dir)
        
        training_data = CustomImageDataset(config.instance_dir, config.resolution)
        train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=config.batch_size) # shuffle=True, num_workers=4

        model = dvhNet()
        # summary(model, [(1,3,224,224), (1, 3, 4)])
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate) # weight_decay=1e-5

        epochs = 2
        for epoch_idx in range(epochs):
            print(f"-------------------------------\nEpoch {epoch_idx+1}")
            loss = train_step(train_dataloader, model, loss_fn, optimizer)
            writer.add_scalar("Loss/train", loss, epoch_idx)
            if epoch_idx % 100 == 0:
                torch.save(model.state_dict(), f'{flags.save_ckpt_dir}dvhNet_weights_{epoch_idx+1}.pth')
        torch.save(model.state_dict(), f'{flags.save_ckpt_dir}dvhNet_weights_{epochs}.pth')
            
        writer.flush()
        writer.close()

        # TODO: test()
        print("################# Done #################")

    elif flags.mode=="test":
        if not flags.load_ckpt_dir:
            sys.exit("ERROR: Checkpoint directory needed for test mode")

        flags.load_ckpt_dir += '/' if flags.load_ckpt_dir.endswith("/") else ''
        ckpt_fps = os.listdir(flags.load_ckpt_dir)
        ckpt_fps.sort()
        ckpt_fp = os.path.join(flags.load_ckpt_dir,ckpt_fps[-1])
        print("ckpt_fp", ckpt_fp)

        model = dvhNet()
        model.load_state_dict(torch.load(ckpt_fp))
        test_data = CustomImageDataset(config.instance_dir, config.resolution)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size) # shuffle=True, num_workers=4
        loss_fn = nn.BCELoss()
        test(test_dataloader, model, loss_fn)
        # model.eval()

