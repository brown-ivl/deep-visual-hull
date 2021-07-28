#!python3


import os, sys
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
import wget
from torchinfo import summary

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

trainWriter = SummaryWriter(os.path.join("runs", "train"))  # default= ./runs/
evalWriter = SummaryWriter(os.path.join("runs", "eval"))
flags = None

def log(message):
    print(message)
    log_fp = str(Path(flags.save_dir, "log.txt").resolve()) # Path.resolve(): resolve symlinks and eliminate ".."
    with open(log_fp, "a") as file: # a: file created if not exist, append not overwrite
        file.write(f"{message}\n")
def print_progress_bar(iteration, total, epoch, total_epochs, loss, decimals=1, length=50, fill='=', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        epoch       - Required  : current epoch string (Int)
        total_epochs- Required  : total number of epochs (Int)
        loss        - Required  : loss (Tensor)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '>' + '-' * (length - filledLength)
    print(f'\rEpoch {str(epoch + 1)}/{str(total_epochs)}: |{bar}| {percent}% Complete, Loss: {str(loss)}',
          end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def train_step(dataloader, model, loss_fn, optimizer, epoch, total_epochs):
    """train operations for one epoch"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    size = len(dataloader.dataset)
    epochLoss = 0
    for batch_idx, (images, points, y) in enumerate(dataloader):
        images, points, y = images.to(device), points.to(device), y.to(device)
        pred = model(images.float(), points.float())  # predicts on the batch of training data
        reshaped_pred = pred.transpose(1, 2)  # (batch_size, T=8, 1)
        try:
            reshaped_pred = reshaped_pred.reshape(
                (-1, config.resolution, config.resolution, config.resolution))
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
        loss, current = loss.item(), batch_idx * len(images)  # (batch size)
        print_progress_bar(batch_idx, len(size), epoch, total_epochs, loss)

    return epochMeanLoss


def visualize_predictions(pred, name, point_centers, after_epoch, threshold=0.5):
    """Visualize predictions and save binvox files"""
    ones = torch.ones(pred.shape)
    zeros = torch.zeros(pred.shape)

    save_dir = flags.save_dir
    if after_epoch:
        save_dir = str(Path(flags.save_dir, f"e{after_epoch}").resolve())
        if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
    log(f"\t\t{save_dir} - {name}: visualization point_cloud.shape={pred.shape}")
    voxel = torch.where(pred.cpu()>=threshold,ones,zeros)
    voxel_fp = str(Path(save_dir, f"{name}_voxel_grid.jpg").resolve())
    util.draw_voxel_grid(voxel, to_show=False, to_disk=True, fp=voxel_fp)
    binvox_fp = str(Path(save_dir, f"{name}_voxel_grid.binvox").resolve())
    util.save_to_binvox(voxel, config.resolution, binvox_fp)


def test(dataloader, model, loss_fn, after_epoch=None):
    """ test or validation function for one epoch
    model: with loaded checkpoint or trained parameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testLosses = []
    size = len(dataloader.dataset)
    IoU_table = {}
    for batch_idx, (images, points, y) in enumerate(dataloader):
        images, points, y = images.to(device), points.to(device), y.to(device)  # points: (batch_size, 3, T)
        with torch.no_grad():
            pred = model(images.float(), points.float())  # (batch_size, 1, T=16**3=4096)
        try:
            reshaped_pred = pred.transpose(1, 2).reshape(
                (-1, config.resolution, config.resolution, config.resolution)) # pred points correspond to voxel centers
            testLosses.append(loss_fn(reshaped_pred.float(), y.float()).item())
        except:
            continue

        # current = (batch_idx + 1) * len(images)  # len(images)=batch size
        epochMeanLoss = np.mean(np.asarray(testLosses))
        # if batch_idx % 20 == 0:
        #     log(f"\t[Test/Val] Batch={batch_idx + 1}: Data = [{current:>5d}/{size:>5d}] |  Running Mean Test/Val Loss = {epochMeanLoss:>7f}")
        # if batch_idx in [0, 1]:
        #     for idx, pred in enumerate(reshaped_pred):# for each voxel grid prediction in batch
        #         visualize_predictions(pred, f"b{batch_idx}_{idx}", points[idx], after_epoch)

        for pred, yy in zip(reshaped_pred, y):
            IoU = util.cal_IoU(pred, yy)
            IoU_table[(pred, yy)] = IoU
        log(batch_idx)
        
    IoU_table = sorted(IoU_table.items(), key =lambda x:x[1], reverse=True)
    for i in range(0,size,100):
        log(IoU_table[i][1])
        visualize_predictions(IoU_table[i][0][0], f"top{i+1}_pred", points[0], after_epoch)
        visualize_predictions(IoU_table[i][0][1], f"top{i+1}_ground_truth", points[0], after_epoch)
    
    return epochMeanLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train",
                        help="One of 'train' or 'test'.")
    parser.add_argument('--save_dir', type=str, default='/save',
                        help="The directory to store the .pth model files and val/test visualization images.")
    parser.add_argument('--load_ckpt_dir', type=str,
                        help="The directory to load .pth model files from. Required for test mode; used for resuming "
                             "training for train mode.")
    parser.add_argument('--num_epoches', type=int, default=2,
                        help="Number of epoches to train for.")
    parser.add_argument('--load_vgg', type=str, nargs='?', const='vgg16_bn-6c64b313.pth',
                        help="Path to pre-trained VGG-16 file.")
    flags, unparsed = parser.parse_known_args()

    # Settings shared across train and test
    loss_fn = nn.BCELoss()
    model = DvhNet()
    if flags.mode == "train":
        log("\n################# Train Mode #################")

        # Initialize model and load checkpoint if passed in

        if torch.cuda.is_available():
            model.cuda()
        startEpoch = 1  # inclusive
        if flags.load_vgg:

            flags.load_vgg = os.path.abspath(flags.load_vgg)
            if not os.path.exists(flags.load_vgg) and not os.path.exists('vgg16_bn-6c64b313.pth'):
                wget.download(
                    "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth")
                flags.load_vgg = os.path.abspath('vgg16_bn-6c64b313.pth')

            log(f"Loading pre-trained VGG-16 file:{flags.load_vgg}")

            if torch.cuda.is_available():
                model.encoder.load_state_dict(torch.load(flags.load_vgg), strict=False)
            else:
                model.encoder.load_state_dict(torch.load(flags.load_vgg,map_location=torch.device('cpu')),strict=False)

            for param in model.encoder.parameters():
                param.requires_grad = False

            flags.save_dir = util.create_checkpoint_directory(flags.save_dir)

        if flags.load_ckpt_dir:
            checkpoint_path = util.get_checkpoint_fp(flags.load_ckpt_dir)
            log(f"Loading latest checkpoint filepath:{checkpoint_path}")
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(checkpoint_path))
            else:
                model.load_state_dict(torch.load(checkpoint_path),map_location=torch.device('cpu'))
            startEpoch = int(checkpoint_path[checkpoint_path.rindex("_") + 1:-4]) + 1
            flags.save_dir = flags.load_ckpt_dir
        else:
            flags.save_dir = util.create_checkpoint_directory(flags.save_dir)
        log(f"save_dir={flags.save_dir}")
        log(f"config.visualize_threshold={config.visualize_threshold}")

        # summary(model, [(1, 3, 224, 224), (1, 3, 4)])
        # Set up data
        train_data = DvhShapeNetDataset(config.train_dir, config.resolution, single_object=config.is_single_instance)
        train_data = nc.SafeDataset(train_data)
        if len(train_data) == 0: sys.exit(f"ERROR: train data not found at {config.train_dir}")
        log(f"Created train_data DvhShapeNetDataset from {config.train_dir}: {len(train_data)} images")
        train_dataloader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=config.batch_size)
        val_data = DvhShapeNetDataset(config.test_dir, config.resolution,  single_object=config.is_single_instance)
        val_data = nc.SafeDataset(val_data)
        if len(val_data) == 0: sys.exit(f"ERROR: val data not found at {config.test_dir}")
        log(f"Created val_data DvhShapeNetDataset from {config.test_dir}: {len(val_data)} images")
        val_dataloader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=config.batch_size)  # shuffle=True, num_workers=4

        # Train and Validate
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # weight_decay=1e-5
        log(f"Training for epochs {startEpoch}-{startEpoch + flags.num_epoches - 1} ({flags.num_epoches} epoches)")
        for epoch_idx in range(startEpoch, startEpoch + flags.num_epoches):
            log(f"-------------------------------\nEpoch {epoch_idx}")
            epochMeanLoss = train_step(train_dataloader, model, loss_fn, optimizer, epoch_idx, startEpoch + flags.num_epoches)
            trainWriter.add_scalar("Loss", epochMeanLoss, global_step=epoch_idx)
            if epoch_idx % 100 == 0:
                torch.save(model.state_dict(), f'{flags.save_dir}dvhNet_weights_{epoch_idx}.pth')
                testEpochMeanLoss = test(val_dataloader, model, loss_fn, after_epoch=epoch_idx)
                evalWriter.add_scalar("Loss", testEpochMeanLoss, global_step=epoch_idx)
        torch.save(model.state_dict(), f'{flags.save_dir}dvhNet_weights_{startEpoch + flags.num_epoches - 1}.pth')
        testEpochMeanLoss = test(val_dataloader, model, loss_fn, after_epoch=startEpoch + flags.num_epoches - 1)
        evalWriter.add_scalar("Loss", testEpochMeanLoss, global_step=epoch_idx)

        trainWriter.flush()
        trainWriter.close()
        evalWriter.flush()
        evalWriter.close()
        log("################# Done #################\n")


    elif flags.mode == "test":
        log("\n################# Test Mode #################")
        if not flags.load_ckpt_dir:
            sys.exit("ERROR: Checkpoint directory needed for test mode. Use '--load_ckpt_dir' flag")
        flags.save_dir = flags.load_ckpt_dir
        log(f"save_dir={flags.save_dir}")
        log(f"config.visualize_threshold={config.visualize_threshold}")


        if torch.cuda.is_available():
            model.cuda()
        checkpoint_path = util.get_checkpoint_fp(flags.load_ckpt_dir)
        log(f"Loading latest checkpoint filepath: {checkpoint_path}")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        test_data = DvhShapeNetDataset(config.test_dir, config.resolution)
        test_data = nc.SafeDataset(test_data)
        if len(test_data) == 0: sys.exit(f"ERROR: test data not found at {config.test_dir}")
        log(f"Created test_data DvhShapeNetDataset from {config.test_dir}: {len(test_data)} images")
        test_dataloader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=config.batch_size, shuffle=True)
        testEpochMeanLoss = test(test_dataloader, model, loss_fn)

        log("################# Done #################\n")
