import numpy as np
import torch
import skimage
import trimesh
from tk3dv.nocstools import datastructures as nocs_ds
import cv2
from pyntcloud.structures import VoxelGrid
import matplotlib.pyplot as plt

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
    batch_size = 64
    points = decoder(latent, torch.split(grid['grid_points'],batch_size,dim=0)).detach().cpu().numpy() # TODO
    #  (batch_size, 3, T) -> (batch_size, 1, T)

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
    # verts=(V, 3) array=spatial coordinates for V unique mesh vertices
    # faces=(F, 3) array=define triangular faces via referencing vertex indices from verts (sometimes based on the order -> normal)
    # normals=(V, 3) array=normal direction at each vertex, as calculated from the data.
    # values=(V, ) array=gives a measure for the maximum value of the data in the local region near each vertex. Used to apply a colormap to the mesh.
    verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])
    meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
    return {"mesh_export": meshexport}

def read_nocs_map(path):
    nocs_map = cv2.imread(path, -1)
    nocs_map = nocs_map[:, :, :3]  # Ignore alpha if present
    nocs_map = cv2.cvtColor(nocs_map, cv2.COLOR_BGR2RGB)
    return nocs_map

def nocs2voxel(nocs_list, resolution = 16):
    ''' Turns a tuple of NOCS maps into a binary voxel grid 
    parameters:
    nocs_list: a list of nocs_map returned by read_nocs_map
    '''
    nocs_pc = []
    for nocs_map in nocs_list: # nocs_map: colored image read in from filepath
        nocs = nocs_ds.NOCSMap(nocs_map)
        nocs_pc.append(nocs.Points) # (x, y, z) in NOCS
    nocs_pc = np.concatenate(nocs_pc, axis=0)

    point_set = nocs_ds.PointSet3D()
    point_set.appendAll(nocs_pc)
    points = np.array(point_set.Points)

    print(points.shape)
    binary_voxel_grid = VoxelGrid(points=points, n_x=resolution, n_y=resolution, n_z=resolution)
    binary_voxel_grid.compute()
    return binary_voxel_grid.get_feature_vector(mode="binary") # [n_x, n_y, n_z] ndarray 

def draw_voxel_grid(binary_voxel_grid):
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(binary_voxel_grid, edgecolor='k')
    plt.show()


# # TEST VOXELIZATION #
# if __name__ == "__main__":
#     nocs_maps = []
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     nocs_maps.append(read_nocs_map("INSERT PATH HERE"))
#     draw_voxel_grid(nocs2voxel(nocs_maps))

# TODO: path reading