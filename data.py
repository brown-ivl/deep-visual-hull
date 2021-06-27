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
from utils.util import get_image, nocs2voxel, calculate_voxel_centers, img_path2numpy

COLOR_IMAGE_FILE_PATH_PATTERN = "*Color_00*"
NOCS_MAP_FILE_PATH_PATTERN = "*NOX*"


class DvhShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, resolution):
        self.voxel_centers = calculate_voxel_centers(resolution)
        self.resolution = resolution
        self.object_id_2_voxel_grid = dict()
        self.image_paths = glob.glob(f"{dir_path}/*/{COLOR_IMAGE_FILE_PATH_PATTERN}")

    def __len__(self):
        return len(self.image_paths)

    def get_voxel_grid_for_object(self, object_path):
        nocs_paths = list(map(str, list(pathlib.Path(object_path).glob(NOCS_MAP_FILE_PATH_PATTERN))))
        nocs_maps = list(map(img_path2numpy, nocs_paths))
        print(nocs_paths)
        return nocs2voxel(nocs_maps, self.resolution)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        parent_dir = pathlib.Path(image_path).parent

        object_id = str(parent_dir.stem)
        category_id = str(parent_dir.parent.stem)

        if object_id not in self.object_id_2_voxel_grid:
            self.object_id_2_voxel_grid[object_id] = self.get_voxel_grid_for_object(str(parent_dir))

        return get_image(image_path), self.voxel_centers.detach().clone(), self.object_id_2_voxel_grid[object_id]
