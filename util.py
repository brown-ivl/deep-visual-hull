import numpy as np
import torch
import skimage
import trimesh
import tqdm

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