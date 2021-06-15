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
            print(f"Train loss: {loss:>7f}") # [{current:>5d}/{size:>5d}]

    return loss

def test(dataloader, model, loss_fn, threshold=0.5, device='cpu', after_epoch=None):
    '''model: with loaded checkpoint or trained parameters'''
    testLosses = []
    objpointcloud = [] # append points from each image-occupancy pair together for one visualization per object
    for batch_idx, (images, points, y) in enumerate(dataloader):
        print("\t----", batch_idx)
        images, points, y = images.to(device), points.to(device), y.to(device)  # points: (batch_size, 3, T)
        pred = model(images.float(), points.float()) # (batch_size, 1, T=16**3=4096)
        # print("\tpred=", pred)
        reshaped_pred = pred.transpose(1, 2).reshape((config.batch_size, config.resolution, config.resolution, config.resolution))
        testLosses.append(loss_fn(reshaped_pred.float(), y.float()).item())
        print(f"\tTest loss: {np.mean(np.asarray(testLosses)):>7f}")

        ## convert prediction to point cloud, then to voxel grid
        indices = torch.nonzero(pred>threshold, as_tuple=True) # tuple of 3 tensors, each the indices of 1 dimensino
        pointcloud = points[indices[0], :, indices[2]].tolist()
        print("\tpointcloud.shape=", np.array(pointcloud).shape)
        objpointcloud += pointcloud # array of [x,y,z] where pred > threshold

    objpointcloud = np.array(objpointcloud)
    print("objpointcloud.shape=", objpointcloud.shape)
    if len(objpointcloud) != 0:
        voxel = util.pointcloud2voxel(objpointcloud, config.resolution)
        voxel_fp = f"{flags.save_dir}voxel_grid_e{after_epoch}.jpg" if after_epoch else f"{flags.load_ckpt_dir}voxel_grid.jpg" 
        util.draw_voxel_grid(voxel, to_show=False, to_disk=True, fp=voxel_fp)


if __name__ == "__main__":
    timestamp = str(int(time.time()))
    print(f"################# Timestamp = {timestamp} #################")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help="One of 'train' or 'test'")
    parser.add_argument('--save_dir', type=str, default=config.save_dir, 
        help="The directory to store the .pth model files and val/test visualization images. If load_ckpt_dir also passed in, then overriden.")
    parser.add_argument('--load_ckpt_dir', type=str, help="The directory to load .pth model files from, required for test mode")
    flags, unparsed = parser.parse_known_args()
    
    if flags.mode=="train":
        print("TRAIN mode")
        if flags.load_ckpt_dir:
            flags.load_ckpt_dir+= '' if flags.load_ckpt_dir.endswith("/") else '/'
            flags.save_dir = flags.load_ckpt_dir
            # TODO: load checkpoint, extract the epoch number, continue training and saving from there (epochidx in range(oldepoch, oldepoch+epohces))
        else:
            flags.save_dir += f'{timestamp}/' if flags.save_dir.endswith("/") else f'{timestamp}/'
        if os.path.exists(flags.save_dir) == False:
            os.makedirs(flags.save_dir)
        print("save_dir=",flags.save_dir)
        
        training_data = CustomImageDataset(config.instance_dir, config.resolution)
        train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=config.batch_size) # shuffle=True, num_workers=4

        model = dvhNet()
        # summary(model, [(1,3,224,224), (1, 3, 4)])
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate) # weight_decay=1e-5

        epochs = 5
        for epoch_idx in range(epochs):
            print(f"-------------------------------\nEpoch {epoch_idx+1}")
            loss = train_step(train_dataloader, model, loss_fn, optimizer)
            writer.add_scalar("Loss/train", loss, epoch_idx)
            if epoch_idx % 100 == 0:
                torch.save(model.state_dict(), f'{flags.save_dir}dvhNet_weights_{epoch_idx+1}.pth')
                test(train_dataloader, model, loss_fn, after_epoch=epoch_idx+1)
        torch.save(model.state_dict(), f'{flags.save_dir}dvhNet_weights_{epochs}.pth')
        test(train_dataloader, model, loss_fn, after_epoch=epochs)

        writer.flush()
        writer.close()

        print("################# Done #################")


    elif flags.mode=="test":
        print("TEST mode")
        if not flags.load_ckpt_dir:
            sys.exit("ERROR: Checkpoint directory needed for test mode")

        flags.load_ckpt_dir += '/' if flags.load_ckpt_dir.endswith("/") else ''
        ckpt_fps = os.listdir(flags.load_ckpt_dir)
        ckpt_fps = [f for f in ckpt_fps if f.endswith(".pth")]
        ckpt_fps.sort()
        if len(ckpt_fps)==0:
            sys.exit("ERROR: cannot find checkpint files in load_ckpt_dir")
        ckpt_fp = os.path.join(flags.load_ckpt_dir,ckpt_fps[-1])
        print("Checkpoint filepath:", ckpt_fp)

        model = dvhNet()
        model.load_state_dict(torch.load(ckpt_fp))
        test_data = CustomImageDataset(config.instance_dir, config.resolution)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size) # shuffle=True, num_workers=4
        loss_fn = nn.BCELoss()
        test(test_dataloader, model, loss_fn)

        print("################# Done #################")

