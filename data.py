import os
import torch
from torchvision.transforms import ToTensor
from util import calculate_voxel_centers
import binvox_rw as binvox
import cv2
import pathlib
import torch
import glob
from util import get_image, nocs2voxel, calculate_voxel_centers, img_path2numpy

"""
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, instance_dir, resolution, transform=None, target_transform=None):
        self.instance_dir = instance_dir
        instance_fp = os.listdir(instance_dir)
        instance_fp.sort()
        self.image_paths = [os.path.join(instance_dir, f) for f in instance_fp if 'Color_00' in f]
        self.voxel_grid_paths = [os.path.join(instance_dir, f) for f in instance_fp if 'voxel_grid' in f]
        self.transform = transform
        self.target_transform = target_transform
        self.points = calculate_voxel_centers(resolution)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = get_image(self.image_paths[idx])
        with open(self.voxel_grid_paths[idx], "rb") as f:
            voxels = binvox.read_as_3d_array(f)
            voxel_grid = voxels.data.astype(float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            voxel_grid = self.target_transform(voxel_grid)
        return image[:3, :, :] , self.points.detach().clone(), voxel_grid
"""

COLOR_IMAGE_FILE_PATH_PATTERN = "*Color_00*"
NOCS_MAP_FILE_PATH_PATTERN = "*NOX*"


class DvhShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, resolution, amount_of_data=1.0, overfitting=False):
        self.amount_of_data = amount_of_data
        self.voxel_centers = calculate_voxel_centers(resolution)
        self.resolution = resolution
        self.object_id_2_voxel_grid = dict()

        self.image_paths = glob.glob(f"{dir_path}/{COLOR_IMAGE_FILE_PATH_PATTERN}") if overfitting is True \
            else glob.glob(f"{dir_path}/*/{COLOR_IMAGE_FILE_PATH_PATTERN}")

    def __len__(self):
        return int(len(self.image_paths) * self.amount_of_data)

    def get_voxel_grid_for_object(self, object_path):
        nocs_paths = list(map(str, list(pathlib.Path(object_path).glob(NOCS_MAP_FILE_PATH_PATTERN))))
        nocs_maps = list(map(img_path2numpy, nocs_paths))
        return nocs2voxel(nocs_maps, self.resolution)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        parent_dir = pathlib.Path(image_path).parent

        object_id = str(parent_dir.stem)
        category_id = str(parent_dir.parent.stem)

        if object_id not in self.object_id_2_voxel_grid:
            self.object_id_2_voxel_grid[object_id] = self.get_voxel_grid_for_object(str(parent_dir))

        return get_image(image_path), self.voxel_centers.detach().clone(), self.object_id_2_voxel_grid[object_id]
