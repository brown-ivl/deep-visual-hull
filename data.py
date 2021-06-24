import pathlib
import torch
import glob
from util import get_image, nocs2voxel, calculate_voxel_centers, imgpath2numpy


class DvhObject3d:
    def __init__(self, instance_dir_path, resolution):
        self.images = list(map(get_image, list(pathlib.Path(instance_dir_path).glob('*Color*'))))
        nocs_maps = list(map(imgpath2numpy, list(pathlib.Path(instance_dir_path).glob('*NOX*'))))
        self.resolution = resolution
        self.voxel_grid = nocs2voxel(nocs_maps, self.resolution)
        self.points = calculate_voxel_centers(self.resolution)

    def __iter__(self):
        for image in self.images:
            yield image, self.points.detach().clone(), self.voxel_grid


class DvhShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, resolution):
        directories = glob.glob(f"{dir_path}/*/") # instance directories
        self.num_objects = len(directories)
        self.directories = iter(directories)
        self.current_object = None
        self.resolution = resolution

    def __len__(self):
        return self.num_objects

    def __getitem__(self, idx):
        # try:
        #     return next(self.current_object)
        # except:
        #     self.current_object = iter(DvhObject3d(next(self.directories), self.resolution))
        #     return next(self.current_object)
        if self.current_object is None:
            self.current_object = iter(DvhObject3d(next(self.directories), self.resolution))
        return next(self.current_object)
