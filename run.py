import argparse
import os
import sys
import nonechucks as nc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import config
import utils.util as util
from data import DvhShapeNetDataset
from models.DvhNet import DvhNet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

writer = SummaryWriter()  # output to ./runs/ directory by default.
flags = None


def train_step(dataloader, model, loss_fn, optimizer):
    """train operations for one epoch"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    size = len(dataloader.dataset)  # number of samples = 2811
    epochLoss = 0
    for batch_idx, (images, points, y) in enumerate(dataloader):
        images, points, y = images.to(device), points.to(device), y.to(device)
        pred = model(images.float(), points.float())  # predicts on the batch of training data
        reshaped_pred = pred.transpose(1, 2)  # (batch_size, T=8, 1)
        try:
            reshaped_pred = reshaped_pred.reshape(
                (config.batch_size, config.resolution, config.resolution, config.resolution))
        except:
            continue
        loss = loss_fn(reshaped_pred.float(), y.float())  # compute prediction error
        # Backpropagation of predication error
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epochLoss += loss.item()
        epochMeanLoss = epochLoss / (batch_idx + 1)
        current = (batch_idx + 1) * len(images)  # len(images)=batch size
        print(f"\tBatch={batch_idx + 1}: Data = [{current:>5d}/{size:>5d}] |  Mean Train Loss = {epochMeanLoss:>7f}")
    return epochMeanLoss


def visualize_predictions(pred, name, point_centers, threshold=0.5):
    indices = torch.nonzero(pred > threshold, as_tuple=True)  # tuple of 3 tensors, each the indices of 1 dimension
    print(indices.shape)
    pointcloud = point_centers[indices[0], :,
                 indices[2]].tolist()  # QUESTION: output pred same order as input points? Result of loss function?
    if len(pointcloud) != 0:
        voxel = util.point_cloud2voxel(pointcloud, config.resolution)
        voxel_fp = str(Path.joinpath(flags.save_dir, f"{name}_voxel_grid.jpg"))
        util.draw_voxel_grid(voxel, to_show=False, to_disk=True, fp=voxel_fp)
        binvox_fp = str(Path.joinpath(flags.save_dir, f"{name}_voxel_grid.binvox"))
        util.save_to_binvox(voxel, config.resolution, binvox_fp)


def test(dataloader, model, loss_fn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """model: with loaded checkpoint or trained parameters"""
    testLosses = []
    size = len(dataloader.dataset)  # number of samples = 2811
    epochLoss = 0

    for batch_idx, (images, points, y) in enumerate(dataloader):
        images, points, y = images.to(device), points.to(device), y.to(device)  # points: (batch_size, 3, T)
        with torch.no_grad():
            pred = model(images.float(), points.float())  # (batch_size, 1, T=16**3=4096)
        try:
            reshaped_pred = pred.transpose(1, 2).reshape(
                (config.batch_size, config.resolution, config.resolution, config.resolution))
            testLosses.append(loss_fn(reshaped_pred.float(), y.float()).item())
        except:
            continue

        loss = loss_fn(reshaped_pred.float(), y.float())  # compute prediction error
        epochLoss += loss.item()
        epoch_mean_loss = epochLoss / (batch_idx + 1)

        for idx, pred in enumerate(reshaped_pred):
            visualize_predictions(pred, f"{batch_idx}_{idx}", points[idx])

        current = (batch_idx + 1) * len(images)  # len(images)=batch size
        print(f"\tBatch={batch_idx + 1}: Data = [{current:>5d}/{size:>5d}] |  Mean Train Loss = {epoch_mean_loss:>7f}")

    print(f"[Test/Val] Mean Loss = {np.mean(np.asarray(testLosses)):>7f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train",
                        help="One of 'train' or 'test'.")
    parser.add_argument('--save_dir', type=str, default=config.save_dir,
                        help="The directory to store the .pth model files and val/test visualization images.")
    parser.add_argument('--load_ckpt_dir', type=str,
                        help="The directory to load .pth model files from. Required for test mode; used for resuming "
                             "training for train mode.")
    parser.add_argument('--num_epoches', type=int, default=5,
                        help="Number of epoches to train for.")
    flags, unparsed = parser.parse_known_args()

    # Settings shared across train and test
    loss_fn = nn.BCELoss()

    if flags.mode == "train":
        print("Train Mode")

        # Initialize model and load checkpoint if passed in
        model = DvhNet()
        if torch.cuda.is_available():
            model.cuda()
        startEpoch = 1  # inclusive
        if flags.load_ckpt_dir:
            checkpoint_path = util.get_checkpoint_fp(flags.load_ckpt_dir)
            print("Loading latest checkpoint filepath:", checkpoint_path)
            model.load_state_dict(torch.load(checkpoint_path))
            startEpoch = int(checkpoint_path[checkpoint_path.rindex("_") + 1:-4]) + 1
            flags.save_dir = flags.load_ckpt_dir
        else:
            flags.save_dir = util.create_checkpoint_directory(flags.save_dir)

        # Set up data
        train_data = DvhShapeNetDataset(config.train_dir, config.resolution)
        train_data = nc.SafeDataset(train_data)
        train_dataloader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=config.batch_size)

        val_data = DvhShapeNetDataset(config.test_dir, config.resolution)
        val_data = nc.SafeDataset(val_data)
        val_dataloader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=config.batch_size)  # shuffle=True, num_workers=4

        # Train and Validate
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # weight_decay=1e-5
        print(f"Training for epochs {startEpoch}-{startEpoch + flags.num_epoches - 1} ({flags.num_epoches} epoches)")
        for epoch_idx in range(startEpoch, startEpoch + flags.num_epoches):
            print(f"-------------------------------\nEpoch {epoch_idx}")
            epochMeanLoss = train_step(train_dataloader, model, loss_fn, optimizer)
            print(f"Epoch Mean Train Loss={epochMeanLoss:>7f}")
            writer.add_scalar("Loss/train", epochMeanLoss, global_step=epoch_idx)
            if epoch_idx % 50 == 0:
                torch.save(model.state_dict(), f'{flags.save_dir}dvhNet_weights_{epoch_idx}.pth')
                test(val_dataloader, model, loss_fn, after_epoch=epoch_idx)
        torch.save(model.state_dict(), f'{flags.save_dir}dvhNet_weights_{startEpoch + flags.num_epoches - 1}.pth')
        test(val_dataloader, model, loss_fn, after_epoch=startEpoch + flags.num_epoches - 1)

        writer.flush()
        writer.close()
        print("################# Done #################")


    elif flags.mode == "test":
        print("Test Mode")
        if not flags.load_ckpt_dir:
            sys.exit("ERROR: Checkpoint directory needed for test mode. Use '--load_ckpt_dir' flag")

        checkpoint_path = util.get_checkpoint_fp(flags.load_ckpt_dir)
        print("Loading latest checkpoint filepath:", checkpoint_path)
        model = DvhNet()
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        test_data = DvhShapeNetDataset(config.test_dir, config.resolution)
        test_data = nc.SafeDataset(test_data)
        test_dataloader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=config.batch_size, shuffle=True)
        if len(test_data) == 0: sys.exit(f"ERROR: test data not found at {config.test_dir}")
        print(f"Created test_data DvhShapeNetDataset from {config.test_dir}: {len(test_data)} images")
        # shuffle=True, num_workers=4
        test(test_dataloader, model, loss_fn)

        print("################# Done #################")
