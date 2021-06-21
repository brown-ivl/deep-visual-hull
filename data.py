import pathlib
import torch
import glob
from util import get_image, nocs2voxel, calculate_voxel_centers, imgpath2numpy
import config


class DvhObject3d:
    def __init__(self, nocs_dir_path):
        self.images = list(map(get_image, list(pathlib.Path(nocs_dir_path).glob('*Color*'))))
        nocs_maps = list(map(imgpath2numpy, list(pathlib.Path(nocs_dir_path).glob('*NOX*'))))
        self.voxel_grid = nocs2voxel(nocs_maps)

    def __iter__(self):
        for image in self.images:
            yield image[:3, :, :], calculate_voxel_centers(config.resolution).detach().clone(), self.voxel_grid


class DvhShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        directories = glob.glob(f"{dir_path}/*/")
        self.num_objects = len(directories)
        self.directories = iter(directories)
        self.current_object = None

    def __len__(self):
        return self.num_objects

    def __getitem__(self, idx):
        try:
            return next(self.current_object)
        except:
            self.current_object = iter(DvhObject3d(next(self.directories)))
            return next(self.current_object)
