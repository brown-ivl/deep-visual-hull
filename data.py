import pathlib
import torch
import glob
from util import get_image, nocs2voxel, calculate_voxel_centers, img_path2numpy

COLOR_IMAGE_FILE_PATH_PATTERN = "*Color_00*"
NOCS_MAP_FILE_PATH_PATTERN = "*NOX*"


class DvhShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, resolution, amount_of_data=1.0, single_object=False):
        self.amount_of_data = amount_of_data
        self.voxel_centers = calculate_voxel_centers(resolution)
        self.resolution = resolution
        self.object_id_2_voxel_grid = dict()

        self.image_paths = glob.glob(f"{dir_path}/{COLOR_IMAGE_FILE_PATH_PATTERN}") if single_object is True \
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
