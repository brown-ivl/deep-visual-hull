"""
import pathlib
import torch
import glob
from util import get_image, nocs2voxel, calculate_voxel_centers, img_path2numpy


class DvhShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, resolution):
        self.image_paths = (glob.glob(f"{dir_path}/*/*Color_00*"))
        self.points = calculate_voxel_centers(resolution)
        self.instance_voxel_grids = {}
        self.resolution = resolution

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ''' Loads and returns a sample from the dataset. Constructed for ShapeNet's directory structure'''
        image_path = self.image_paths[idx]
        last = image_path.rfind("/")
        secondlast = image_path.rfind("/", 0, last)
        thirdlast = image_path.rfind("/", 0, secondlast)
        instance_id = image_path[secondlast+1:last]
        if thirdlast>0: # */category/instance/image.png
            category_id = image_path[thirdlast+1:secondlast]
            instancekey = f"{category_id}/{instance_id}"
        else: # */instance/image.png
            instancekey = f"{instance_id}"
        if instancekey not in self.instance_voxel_grids:
            nocs_maps = list(map(img_path2numpy, glob.glob(f"{image_path[:last]}/*NOX*")))
            self.instance_voxel_grids[instancekey] = nocs2voxel(nocs_maps, self.resolution)
        return get_image(image_path), self.points.detach().clone(), self.instance_voxel_grids[instancekey]
"""

import pathlib
import torch
import glob
from util import get_image, nocs2voxel, calculate_voxel_centers, img_path2numpy

COLOR_IMAGE_FILE_PATH_PATTERN = "*Color_00*"
NOCS_MAP_FILE_PATH_PATTERN = "*NOX*00*"


class DvhObject3d:
    def __init__(self, nocs_dir_path: str, resolution: int):
        # image_paths = list(map(str, list(pathlib.Path(nocs_dir_path).glob(COLOR_IMAGE_FILE_PATH_PATTERN))))
        nocs_paths = list(map(str, list(pathlib.Path(nocs_dir_path).glob(NOCS_MAP_FILE_PATH_PATTERN))))
        # self.images = list(map(get_image, image_paths))
        nocs_maps = list(map(img_path2numpy, nocs_paths))
        self.voxel_grid = nocs2voxel(nocs_maps, resolution)


class DvhShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, resolution):
        self.voxel_centers = calculate_voxel_centers(resolution)
        self.directories = glob.glob(f"{dir_path}/*/")
        self.resolution = resolution
        self.directories_to_objects = dict()
        self.image_paths = glob.glob(f"{dir_path}/*/{COLOR_IMAGE_FILE_PATH_PATTERN}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        parent_directory = str(pathlib.Path(image_path).parent.absolute())
        if parent_directory not in self.directories_to_objects:
            self.directories_to_objects[parent_directory] = DvhObject3d(parent_directory, self.resolution)
        obj = self.directories_to_objects[parent_directory]

        return get_image(image_path), self.voxel_centers.detach().clone(), obj.voxel_grid
