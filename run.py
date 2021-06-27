#!python3


import os, sys
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import config
from data import DvhShapeNetDataset
import util

os.environ['KMP_DUPLICATE_LIB_OK']='True'

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, 'models'))
from models.dvhNet import dvhNet


writer = SummaryWriter() # output to ./runs/ directory by default.
flags = None


def train_step(dataloader, model, loss_fn, optimizer, device='cpu'):
    '''train operations for one epoch'''
    size = len(dataloader.dataset) # number of samples = 2811
    epochLoss = 0
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

        epochLoss += loss.item()
        epochMeanLoss = epochLoss/(batch_idx+1)
        current = (batch_idx+1) * len(images) # len(images)=batch size
        print(f"\tBatch={batch_idx+1}: Data = [{current:>5d}/{size:>5d}] |  Mean Train Loss = {epochMeanLoss:>7f}")
    return epochMeanLoss

def test(dataloader, model, loss_fn, threshold=0.5, device='cpu', after_epoch=None):
    '''model: with loaded checkpoint or trained parameters'''
    testLosses = []
    objpointcloud = [] # append points from each image-occupancy pair together for one visualization per object
    for batch_idx, (images, points, y) in enumerate(dataloader):
        images, points, y = images.to(device), points.to(device), y.to(device)  # points: (batch_size, 3, T)
        pred = model(images.float(), points.float()) # (batch_size, 1, T=16**3=4096)
        reshaped_pred = pred.transpose(1, 2).reshape((config.batch_size, config.resolution, config.resolution, config.resolution))
        testLosses.append(loss_fn(reshaped_pred.float(), y.float()).item())
        ## convert prediction to point cloud, then to voxel grid
        indices = torch.nonzero(pred>threshold, as_tuple=True) # tuple of 3 tensors, each the indices of 1 dimension
        pointcloud = points[indices[0], :, indices[2]].tolist() # QUESTION: output pred same order as input points? Result of loss function?
        objpointcloud += pointcloud # array of [x,y,z] where pred > threshold
    objpointcloud = np.array(objpointcloud)
    if len(objpointcloud) != 0:
        voxel = util.pointcloud2voxel(objpointcloud, config.resolution)
        voxel_fp = f"{flags.save_dir}voxel_grid_e{after_epoch}.jpg" if after_epoch else f"{flags.load_ckpt_dir}voxel_grid.jpg"
        util.draw_voxel_grid(voxel, to_show=False, to_disk=True, fp=voxel_fp)
        binvox_fp = f"{flags.save_dir}voxel_grid_e{after_epoch}.binvox" if after_epoch else f"{flags.load_ckpt_dir}voxel_grid.binvox"
        util.save_to_binvox(voxel, config.resolution, binvox_fp)
    print(f"[Test/Val] Mean Loss = {np.mean(np.asarray(testLosses)):>7f} | objpointcloud.shape={objpointcloud.shape}")


if __name__ == "__main__":
    timestamp = str(int(time.time()))
    print(f"\n################# Timestamp = {timestamp} #################")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train",
        help="One of 'train' or 'test'.")
    parser.add_argument('--save_dir', type=str, default=config.save_dir, 
        help="The directory to store the .pth model files and val/test visualization images.")
    parser.add_argument('--load_ckpt_dir', type=str,
        help="The directory to load .pth model files from. Required for test mode; used for resuming training for train mode.")
    parser.add_argument('--num_epoches', type=int, default=5,
        help="Number of epoches to train for.")
    flags, unparsed = parser.parse_known_args()
    
    # Settings shared across train and test
    loss_fn = nn.BCELoss()

    if flags.mode=="train":
        print("TRAIN mode")

        # Check save dir
        flags.save_dir += f'{timestamp}/' if flags.save_dir.endswith("/") else f'{timestamp}/'
        if os.path.exists(flags.save_dir) == False:
            os.makedirs(flags.save_dir)
        print("save_dir=",flags.save_dir)

        # Initiatilize model and load checkpoint if passed in
        model = dvhNet()
        startEpoch = 1 # inclusive
        if flags.load_ckpt_dir:
            ckpt_fp = util.get_checkpoint_fp(flags.load_ckpt_dir)
            print("Loading load_ckpt_dir's latest checkpoint filepath:", ckpt_fp)
            model.load_state_dict(torch.load(ckpt_fp))
            startEpoch = int(ckpt_fp[ckpt_fp.rindex("_")+1:-4])+1
        
        # Set up data
        train_data = DvhShapeNetDataset(config.train_dir, config.resolution)
        if len(train_data) == 0: sys.exit(f"ERROR: training data not found at {config.train_dir}")
        print(f"Created train_data DvhShapeNetDataset from {config.train_dir}: {len(train_data)} images")
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size) # shuffle=True, num_workers=4
        val_data = DvhShapeNetDataset(config.test_dir, config.resolution)
        if len(val_data) == 0: sys.exit(f"ERROR: validation data not found at {config.test_dir}")
        print(f"Created val_data DvhShapeNetDataset from {config.train_dir}: {len(val_data)} images")
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size) # shuffle=True, num_workers=4

        # Train and Validate
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) # weight_decay=1e-5
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate) #momentum=0.9 (like beta), nesterov=True, weight_decay=1e-5
        print(f"Training for epoches {startEpoch}-{startEpoch+flags.num_epoches-1} ({flags.num_epoches} epoches)")
        for epoch_idx in range(startEpoch, startEpoch+flags.num_epoches):
            print(f"-------------------------------\nEpoch {epoch_idx}")
            epochMeanLoss = train_step(train_dataloader, model, loss_fn, optimizer)
            print(f"Epoch Mean Train Loss={epochMeanLoss:>7f}")
            writer.add_scalar("Loss/train", epochMeanLoss, global_step=epoch_idx)
            if epoch_idx % 50 == 0:
                torch.save(model.state_dict(), f'{flags.save_dir}dvhNet_weights_{epoch_idx}.pth')
                test(val_dataloader, model, loss_fn, after_epoch=epoch_idx)
        torch.save(model.state_dict(), f'{flags.save_dir}dvhNet_weights_{startEpoch+flags.num_epoches-1}.pth')
        test(val_dataloader, model, loss_fn, after_epoch=startEpoch+flags.num_epoches-1)

        writer.flush()
        writer.close()
        print("################# Done #################")


    elif flags.mode=="test":
        print("TEST mode")
        if not flags.load_ckpt_dir:
            sys.exit("ERROR: Checkpoint directory needed for test mode. Use '--load_ckpt_dir' flag")
        else:
            flags.load_ckpt_dir+= '' if flags.load_ckpt_dir.endswith("/") else '/'

        ckpt_fp = util.get_checkpoint_fp(flags.load_ckpt_dir)
        print("Loading load_ckpt_dir's latest checkpoint filepath:", ckpt_fp)
        model = dvhNet()
        model.load_state_dict(torch.load(ckpt_fp))

        test_data = DvhShapeNetDataset(config.test_dir, config.resolution)
        if len(test_data) == 0: sys.exit(f"ERROR: test data not found at {config.test_dir}")
        print(f"Created test_data DvhShapeNetDataset from {config.test_dir}: {len(test_data)} images")
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=config.batch_size) # shuffle=True, num_workers=4
        test(test_dataloader, model, loss_fn)

        print("################# Done #################")

