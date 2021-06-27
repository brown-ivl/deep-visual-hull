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


# class DvhObject3d:
#     def __init__(self, nocs_dir_path, resolution):
#         self.images = list(map(get_image, list(pathlib.Path(nocs_dir_path).glob('*Color*'))))
#         nocs_maps = list(map(imgpath2numpy, list(pathlib.Path(nocs_dir_path).glob('*NOX*'))))
#         self.voxel_grid = nocs2voxel(nocs_maps, resolution)
#         self.resolution = resolution

#     def __iter__(self):
#         for image in self.images:
#             yield image[:3, :, :], calculate_voxel_centers(self.resolution).detach().clone(), self.voxel_grid


# class DvhShapeNetDataset(torch.utils.data.Dataset):
#     def __init__(self, dir_path, resolution):
#         directories = glob.glob(f"{dir_path}/*/")
#         self.num_objects = len(directories)
#         self.directories = iter(directories)
#         self.current_object = None
#         self.resolution = resolution

#     def __len__(self):
#         return self.num_objects

#     def __getitem__(self, idx):
#         try:
#             return next(self.current_object)
#         except:
#             self.current_object = iter(DvhObject3d(next(self.directories), self.resolution))
#             return next(self.current_object)