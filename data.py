import pathlib
import torch
import glob
from util import get_image, nocs2voxel, calculate_voxel_centers, imgpath2numpy

class DvhObject3d:
    def __init__(self, nocs_dir_path, resolution):
        self.images = list(map(get_image, list(pathlib.Path(nocs_dir_path).glob('*Color_00*'))))
        nocs_maps = list(map(imgpath2numpy, list(pathlib.Path(nocs_dir_path).glob('*NOX*'))))
        self.voxel_grid = nocs2voxel(nocs_maps, resolution)
        self.resolution = resolution

    def __iter__(self):
        for image in self.images:
            yield image[:3, :, :], calculate_voxel_centers(self.resolution).detach().clone(), self.voxel_grid


class DvhShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, resolution):
        directories = glob.glob(f"{dir_path}/*/")
        self.num_objects = len(directories)
        self.num_images = len((glob.glob(f"{dir_path}/*/*Color*")))
        self.directories = iter(directories)
        self.current_object = None
        self.resolution = resolution

    def __len__(self):
        #return self.num_objects
        return self.num_images

    def __getitem__(self, idx):
        try:
            return next(self.current_object)
        except:
            self.current_object = iter(DvhObject3d(next(self.directories), self.resolution))
            return next(self.current_object)