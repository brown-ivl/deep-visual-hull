import os
import torch
from torchvision.io import read_image
from torchvision.transforms import ToTensor
from util import calculate_voxel_centers
import binvox_rw as binvox

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, instance_dir, resolution, transform=ToTensor(), target_transform=None):
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
        image = read_image(self.image_paths[idx])
        voxel_grid = self.img_labels.iloc[idx, 1]
        with open(self.voxel_grid_paths[idx], "rb") as f:
            voxels = binvox.read_as_3d_array(f)
            label = voxels.data.astype(float)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, self.points.detach().clone(), label
