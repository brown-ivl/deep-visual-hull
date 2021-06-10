import os
import torch
from torchvision.transforms import ToTensor
from util import calculate_voxel_centers
import binvox_rw as binvox
import cv2

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

    def get_image(self, path):
        '''takes an image path and returns a tensor of the shape 3 (num channels) x 224 x 224'''
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        return image

    def __getitem__(self, idx):
        image = self.get_image(self.image_paths[idx])
        with open(self.voxel_grid_paths[idx], "rb") as f:
            voxels = binvox.read_as_3d_array(f)
            voxel_grid = voxels.data.astype(float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            voxel_grid = self.target_transform(voxel_grid)
        return image[:3, :, :] , self.points.detach().clone(), voxel_grid
