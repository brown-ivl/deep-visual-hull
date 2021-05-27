import numpy as np
import torch
import skimage
import trimesh
from tk3dv.nocstools import datastructures as nocs_ds
import cv2
from pyntcloud.structures import VoxelGrid

def get_grid_uniform(resolution):
    x = np.linspace(-1.2,1.2, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.4,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_mesh(decoder, latent, resolution, mc_value):

    grid = get_grid_uniform(resolution)
    points = decoder(latent, torch.split(grid['grid_points'],100000,dim=0)).detach().cpu().numpy()

    print(np.min(points))
    print(np.max(points))
    print(mc_value)

    points = points.astype(np.float64)

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        volume=points.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=mc_value,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])

    meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)

    return {"mesh_export": meshexport}

def read_nocs_map(path):
    nocs_map = cv2.imread(path, -1)
    nocs_map = nocs_map[:, :, :3]  # Ignore alpha if present
    nocs_map = cv2.cvtColor(nocs_map, cv2.COLOR_BGR2RGB)
    return nocs_map

def nocs2voxel(nocs_list, resolution = 16):
    ''' Turns a tuple of NOCS maps into a binary voxel grid '''
    nocs_pc = []
    for nocs_map in nocs_list:
        nocs = nocs_ds.NOCSMap(nocs_map)
        nocs_pc.append(nocs.Points)
    nocs_pc = np.concatenate(nocs_pc, axis=0)

    point_set = nocs_ds.PointSet3D
    point_set.appendAll(nocs_pc)

    binary_voxel_grid = VoxelGrid(point_set.Points, n_x=resolution, n_y=resolution, n_z=resolution).get_feature_vector()

    return binary_voxel_grid