
import os
import sys
import json

import open3d as o3d
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

COLORMAP = 'jet'

def read_config():

	with open('config.json', 'r') as file:
		return json.load(file)

def rotate(x_arr, y_arr, angle):

	"""
	Rotate a point counterclockwise by a given angle around a given origin.

	The angle should be given in radians.
	"""

	ox = 0.0
	oy = 0.0

	x_rot = ox + np.cos(angle) * (x_arr - ox) - np.sin(angle) * (y_arr - oy)
	y_rot = oy + np.sin(angle) * (x_arr - ox) + np.cos(angle) * (y_arr - oy)

	return x_rot, y_rot

if __name__ == '__main__':

	config = read_config()
	
	read_mesh_path = config['data_path'] + "Tanum 89_1/Tanum 89_1 merge"
	read_data_path = config['save_path'] + "Tanum_89_1/Tanum_89_1_merge"


	mesh = o3d.read_triangle_mesh(read_mesh_path + '.stl')
	mesh.compute_vertex_normals()
	mesh.purge()

	vertices = np.asarray(mesh.vertices)
	normals = np.asarray(mesh.vertex_normals)

	transform_data = np.load(read_data_path + '/arrays/transform_meta.npz')
	components = transform_data['components']
	mean = transform_data['mean']
	rotation = transform_data['rotation']

	vertices = np.dot(vertices - mean, components.T)
	normals = np.dot(normals, components.T)

	vertices[:, 0], vertices[:, 1] = rotate(
		vertices[:, 0],
		vertices[:, 1],
		np.radians(rotation)
	)

	normals[:, 0], normals[:, 1] = rotate(
		normals[:, 0],
		normals[:, 1],
		np.radians(rotation)
	)

	data_map = np.load(read_data_path + '/arrays/enhanced_topo_map_object_level.npz')['data']
	data_map = np.nan_to_num(data_map)
	cmap = plt.cm.get_cmap(COLORMAP)
	cnorm = colors.Normalize(vmin=np.nanmin(data_map), vmax=np.nanmax(data_map))
	smap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)

	pix_to_coords_map = np.load(read_data_path + '/arrays/pix_to_coords_map.npz')['data']
	grid_x = pix_to_coords_map[0, :, :]
	grid_y = pix_to_coords_map[1, :, :]

	f = sp.interpolate.interp2d(
		grid_x[0, :],
		grid_y[:, 0],
		data_map,
		kind='linear'
	)
	
	vertex_values = []
	pbar = tqdm(total=len(vertices))
	for v in range(len(vertices)):

		vertex_values.append(f(vertices[v, 0], vertices[v, 1]))
		pbar.update(1)

	pbar.close()

	vertex_colors = [col[0, :3] for col in smap.to_rgba(vertex_values)]

	mesh.vertices = o3d.Vector3dVector(vertices)
	mesh.vertex_normals = o3d.Vector3dVector(normals)
	mesh.vertex_colors = o3d.Vector3dVector(vertex_colors)

	mesh.compute_triangle_normals()

	vis = o3d.Visualizer()
	vis.create_window()
	vis.add_geometry(mesh)
	vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])
	vis.run()
	vis.destroy_window()
