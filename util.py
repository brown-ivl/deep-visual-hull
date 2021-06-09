import numpy as np
import torch
# import skimage
import cv2
import matplotlib.pyplot as plt
# import trimesh
from tk3dv.nocstools import datastructures as nocs_ds
from pyntcloud.structures import VoxelGrid
import os
import binvox_rw as binvox
import config

def get_grid_uniform(resolution):
    x = np.linspace(0, 1, resolution)
    y = x
    z = x
    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float) # [resolution**3 [x, y, z]]
    return {"grid_points": grid_points, "xyz": [x, y, z] } 

def get_mesh(decoder, latent_c, resolution, mc_value):
    grid = get_grid_uniform(resolution)
    batch_size = 64
    points = torch.split(grid['grid_points'],batch_size,dim=0) # TODO: set resolution and reshape to (batch_size, 3, T)
    points = decoder(points, latent_c) # (batch_size, 1, T)
    points = points.detach().cpu().numpy()
    # print(np.min(points))
    # print(np.max(points))
    # print(mc_value)

    points = points.astype(np.float64)

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        volume=points.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=mc_value,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))
    # verts=(V, 3) array=spatial coordinates for V unique mesh vertices
    # faces=(F, 3) array=define triangular faces via referencing vertex indices from verts (sometimes based on the order -> normal)
    # normals=(V, 3) array=normal direction at each vertex, as calculated from the data.
    # values=(V, ) array=gives a measure for the maximum value of the data in the local region near each vertex. Used to apply a colormap to the mesh.
    verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])
    meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
    return {"mesh_export": meshexport}


def read_nocs_map(nocs_paths):
    nocs_images = []
    for nocs_path in nocs_paths:
        nocs_image = cv2.imread(nocs_path)[:, :, :3]  # Ignore alpha if present
        nocs_images.append(cv2.cvtColor(nocs_image, cv2.COLOR_BGR2RGB))
    return nocs_images


def nocs2voxel(nocs_images, resolution):
    ''' Turns a tuple of NOCS maps into a binary voxel grid 
    parameters:
    nocs_images: a list of nocs_image returned by read_nocs_map (2d array of RGB triplets)
    '''
    nocs_pc = []
    for nocs_image in nocs_images:
        nocs_map = nocs_ds.NOCSMap(nocs_image)
        nocs_pc.append(nocs_map.Points) # array of 0-1 [x,y,z] NOCS coordinates where object is observed in rgb images
    nocs_pc = np.concatenate(nocs_pc, axis=0) # array of all ocucpied xyz NOCS coordinates from all nocs_images
    # point_set = nocs_ds.PointSet3D()
    # point_set.appendAll(nocs_pc)
    # points = np.array(point_set.Points)
    points = nocs_pc

    # turns point cloud into voxel grid representation
    binary_voxel_grid = VoxelGrid(points=points, n_x=resolution, n_y=resolution, n_z=resolution)
    binary_voxel_grid.compute()
    bvg_vector = binary_voxel_grid.get_feature_vector(mode="binary") # [n_x, n_y, n_z] ndarray
    return bvg_vector

def calculate_voxel_centers(resolution, min=0, max=1):
    '''returns an array of (x,y,z) coordinates in min-max representing centers of voxel cells
    Example output fo resolution=2:
        tensor([[0.2500, 0.2500, 0.2500],
                [0.2500, 0.2500, 0.7500],
                [0.7500, 0.2500, 0.2500],
                [0.7500, 0.2500, 0.7500],
                [0.2500, 0.7500, 0.2500],
                [0.2500, 0.7500, 0.7500],
                [0.7500, 0.7500, 0.2500],
                [0.7500, 0.7500, 0.7500]])
    After transpose:
        tensor([[0.2500, 0.2500, 0.7500, 0.7500, 0.2500, 0.2500, 0.7500, 0.7500],
                [0.2500, 0.2500, 0.2500, 0.2500, 0.7500, 0.7500, 0.7500, 0.7500],
                [0.2500, 0.7500, 0.2500, 0.7500, 0.2500, 0.7500, 0.2500, 0.7500]])
    '''
    x = np.linspace(min+(max-min)/(resolution*2), max-(max-min)/(resolution*2), resolution)
    y=x
    z=x
    xx, yy, zz = np.meshgrid(x, y, z)
    voxel_centers = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)
    voxel_centers = voxel_centers.transpose(0, 1) # (T, 3) -> (3, T)
    return voxel_centers

def draw_voxel_grid(binary_voxel_grid):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(binary_voxel_grid, edgecolor='k')
    plt.show()

# TEST VOXELIZATION #
if __name__ == "__main__":
    ## For more than 1 object instance:
    # nocs_paths = []
    # dataDir = "shapenetplain_v1/train/02691156/"
    # for (dirpath, dirnames, filenames) in os.walk(dataDir):
    #     for objectDir in dirnames[:1]: # first object directory (example: "a36d00e2f7414043f2b0736dd4d8afe0")
    #         files = os.listdir(os.path.join(dirpath, objectDir))
    #         files = [os.path.join(dirpath, objectDir, f) for f in files if 'NOX' in f]
    #         nocs_paths.extend(files)
    #     break # only one level down

    ## 1 instance
    files = os.listdir(config.instance_dir)
    nocs_paths = [os.path.join(config.instance_dir, f) for f in files if 'NOX' in f]
    frames = list(set([f[6:14] for f in files]))
    nocs_images = read_nocs_map(nocs_paths)
    binary_voxel_grid = nocs2voxel(nocs_images, resolution=config.resolution) # (2,2,2) -> (batch_size, 2,2,2)
    voxel_centers = calculate_voxel_centers(config.resolution) # (3,T=8) -> (batch_size, 3, T=8)
    draw_voxel_grid(binary_voxel_grid)
    # Writing of binvox files
        # for frame in frames:
        #     with open(f"{config.instance_dir}/frame_{frame}_voxel_grid.binvox", "wb") as f:
        #         voxels = binvox.Voxels(np.array(binary_voxel_grid, dtype=bool), [config.resolution, config.resolution, config.resolution], [0,0,0], 1, "xyz")
        #         voxels.write(f)
   