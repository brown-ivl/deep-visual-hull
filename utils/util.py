import glob
import os
import sys
from typing import List
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyntcloud.structures import VoxelGrid
from tk3dv.nocstools import datastructures as nocs_ds
import time
import utils.binvox_rw as binvox_rw
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_voxel_centers(resolution: int, min: int = 0, max: int = 1) -> torch.Tensor:
    """returns an array of (x,y,z) coordinates in min-max representing centers of voxel cells
    Example output of resolution=2:
        tensor([[0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.7500],
                [0.7500, 0.2500, 0.2500],
                [0.7500, 0.2500, 0.7500],
                [0.2500, 0.7500, 0.2500],
                [0.2500, 0.7500, 0.7500],
                [0.7500, 0.7500, 0.2500],
                [0.7500, 0.7500, 0.7500]])
    After transpose:
        tensor([[0.2500, 0.2500, 0.7500, 0.7500, 0.2500, 0.2500, 0.7500, 0.7500],
                [0.2500, 0.2500, 0.2500, 0.2500, 0.7500, 0.7500, 0.7500, 0.7500],
                [0.2500, 0.7500, 0.2500, 0.7500, 0.2500, 0.7500, 0.2500, 0.7500]])
    """
    x = np.linspace(min + (max - min) / (resolution * 2), max - (max - min) / (resolution * 2), resolution)
    y = x
    z = x
    xx, yy, zz = np.meshgrid(x, y, z)
    voxel_centers = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)
    voxel_centers = voxel_centers.transpose(0, 1)  # (T=resolution**3, 3) -> (3, T)
    return voxel_centers.to(device)


def point_cloud2voxel(points, resolution, mode='binary') -> np.ndarray:
    """ Turns a numpy array of xyz coordinates into a voxel grid representation of given resolution"""
    binary_voxel_grid = VoxelGrid(points=points, n_x=resolution, n_y=resolution, n_z=resolution)
    binary_voxel_grid.compute()
    return binary_voxel_grid.get_feature_vector(mode)


def nocs2voxel(nocs_images: List[np.ndarray], resolution: int = config.resolution) -> np.ndarray:
    """ Turns a list of NOCS maps into a binary voxel grid
    parameters:
        nocs_images: a list of nocs_image returned by img_path2numpy (2d array of RGB triplets)
    """
    nocs_pc = []
    for nocs_image in nocs_images:
        try:
            nocs_map = nocs_ds.NOCSMap(nocs_image)
        except:
            cv2.imwrite(f"{os.getcwd()}/error/{get_timestamp()}_error.jpg", nocs_image)
            continue
        nocs_pc.append(nocs_map.Points)
    if len(nocs_pc) > 0:
        nocs_pc = np.concatenate(nocs_pc, axis=0)
    points = nocs_pc

    return point_cloud2voxel(points, resolution)


def draw_voxel_grid(binary_voxel_grid: List[bool], to_show: bool = False, to_disk: bool = False,
                    fp: str = 'voxel_grid.jpg') -> ():
    """visualize binary_voxel_grid, output from point_cloud2voxel()"""
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(binary_voxel_grid, edgecolor='k')
    if to_disk:
        plt.savefig(fp)
    if to_show:
        plt.show()


def get_image(path: str) -> torch.Tensor:
    """takes an image path and returns a tensor of the shape 3 (num channels) x 224 x 224"""
    image = img_path2numpy(path)
    image = cv2.resize(image, (224, 224))
    image = torch.tensor(image)
    # TODO: are we permuting this to swivel color channels, if so get rid of this.
    image = image.permute(2, 0, 1)
    return image.to(device)


def img_path2numpy(path: str) -> np.ndarray:
    image = cv2.imread(path)[:, :, :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def get_checkpoint_fp(directory_path: str) -> str:
    os.path.join(directory_path, "*.pth")
    try:
        return max(glob.glob(os.path.join(directory_path, "*.pth")), key=os.path.getctime)
    except ValueError:
        sys.exit(f"ERROR: cannot find checkpoint files in directory '{directory_path}'")


def get_timestamp() -> str:
    return str(int(time.time()))


def create_checkpoint_directory(directory_path: str) -> str:
    timestamp = get_timestamp()
    checkpoint_dir = os.path.join(directory_path, f'{timestamp}/')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("save_dir=", checkpoint_dir)
    return checkpoint_dir


def save_to_binvox(voxel_grid: List[bool], resolution: int, save_path: str) -> ():
    binvox_ds = binvox_rw.Voxels(
        data=np.array(voxel_grid, dtype=bool),
        dims=[resolution, resolution, resolution],
        translate=[0.0, 0.0, 0.0],
        scale=1.0,
        axis_order='xyz'
    )
    with open(save_path, 'wb') as f:
        binvox_ds.write(f)


def read_binvox(binvox_fp: str) -> np.ndarray:
    with open(binvox_fp, "rb") as f:
        voxels = binvox_rw.read_as_3d_array(f)
    return voxels.data.astype(float)

# if __name__ == "__main__":
# For more than 1 object instance: read nocs maps
# nocs_paths = []
# dataDir = "shapenetplain_v1/train/02691156/"
# for (dirpath, dirnames, filenames) in os.walk(dataDir):
#     for objectDir in dirnames[:1]: # first object directory (example: "a36d00e2f7414043f2b0736dd4d8afe0")
#         files = os.listdir(os.path.join(dirpath, objectDir))
#         files = [os.path.join(dirpath, objectDir, f) for f in files if 'NOX' in f] # nocsmaps
#         nocs_paths.extend(files)
#     break # only one level down


# For 1 instance: read nocs map, draw_voxel_grid, write binvox files
# files = os.listdir(config.instance_dir)
# nocs_paths = [os.path.join(config.instance_dir, f) for f in files if 'NOX' in f]
# nocs_images = read_nocs_map(nocs_paths)
# binary_voxel_grid = nocs2voxel(nocs_images, resolution=config.resolution) # (2,2,2) -> (batch_size, 2,2,2)
# draw_voxel_grid(binary_voxel_grid)

# frames = list(set([f[6:14] for f in files]))
# for frame in frames:
#     save_to_binvox(binary_voxel_grid, config.resolution, f"{config.instance_dir}/frame_{frame}_voxel_grid.binvox")


# Generate voxel centers
# voxel_centers = calculate_voxel_centers(config.resolution) # (3,T=8) -> (batch_size, 3, T=8)


# Visualize a binvox file
# voxel_grid = read_binvox("car_instance/frame_00000000_voxel_grid.binvox")
# print(voxel_grid)
# draw_voxel_grid(voxel_grid, to_show=True, to_disk=True, fp="car_instance/frame_00000000_voxel_grid.jpg")
