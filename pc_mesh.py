
import itertools

import numpy as np
import scipy as sp
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import multiprocessing as mp

from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN, OPTICS
from itertools import repeat, product

if __name__ == '__main__':
	mp.freeze_support()

def visualize_point_clouds(pcd_list):

	vis = o3d.visualization.Visualizer()
	vis.create_window()

	for pcd in pcd_list:
		vis.add_geometry(pcd)

	vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])#np.asarray([0.1, 0.1, 0.1])
	vis.run()
	vis.destroy_window()

def read_mesh(path, down_sample, visualize=False):

	mesh = o3d.io.read_triangle_mesh(path)
	mesh.compute_vertex_normals()

	mesh.remove_duplicated_triangles()
	mesh.remove_duplicated_vertices()

	vertices = np.asarray(mesh.vertices)
	random_idx = np.random.choice(len(mesh.triangles), 100000)
	random_triangles = np.asarray(mesh.triangles)[random_idx]
	
	random_edges = vertices[random_triangles[:, 1]] - vertices[random_triangles[:, 0]]
	mesh_edge_resolution = np.average(np.linalg.norm(random_edges, axis=1))

	print('Nr of Vertices: ' + str(len(mesh.vertices)))
	print('Mesh Edge Resolution: ' + ('%.2f' % mesh_edge_resolution) + ' mm')

	if down_sample['enabled']:

		print('Down Sampling Mesh.')

		'''
		mesh = o3d.simplify_quadric_decimation(
			mesh,
			target_number_of_triangles=100#int(len(mesh.vertices)/2.0)
		)
		'''

		mesh = mesh.simplify_vertex_clustering(
			voxel_size=down_sample['voxel_multiplier'] * mesh_edge_resolution,
			contraction=o3d.geometry.SimplificationContraction.Quadric
		)

		vertices = np.asarray(mesh.vertices)
		random_idx = np.random.choice(len(mesh.triangles), 100000)
		random_triangles = np.asarray(mesh.triangles)[random_idx]
		
		random_edges = vertices[random_triangles[:, 1]] - vertices[random_triangles[:, 0]]
		mesh_edge_resolution = np.average(np.linalg.norm(random_edges, axis=1))

		print('Nr of Vertices: ' + str(len(mesh.vertices)))
		print('Mesh Edge Resolution: ' + ('%.2f' % mesh_edge_resolution) + ' mm')

	if visualize:
		visualize_point_clouds([mesh])

	return mesh, mesh_edge_resolution

def create_point_cloud(mesh, resolution, update_resolution, visualize=False):

	pcd = o3d.geometry.PointCloud()
	pcd.points = mesh.vertices
	pcd.normals = mesh.vertex_normals
	pcd.paint_uniform_color([0.7, 0.7, 0.7])

	# Assume wrong units in coord system if resolution is too low.
	if update_resolution['enabled']:

		print('Updating Resolution.')

		pcd.points = o3d.utility.Vector3dVector(update_resolution['scale_multiplier'] * np.asarray(pcd.points))
		resolution = update_resolution['scale_multiplier'] * resolution

	print('Nr of Points: ' + str(len(pcd.points)))

	if visualize:
		visualize_point_clouds([
			pcd.voxel_down_sample(voxel_size=5.0*resolution)
		])

	return pcd, resolution

def check_noise_init(comp, grid_x, grid_y, resolution):

	global g_comp, g_grid_x, g_grid_y, g_res
	g_comp = comp
	g_grid_x = grid_x
	g_grid_y = grid_y
	g_res = resolution

def check_noise_worker(coords):

	mask = (
		(g_comp[:, 0] >= g_grid_x[coords[0]]) &
		(g_comp[:, 0] <= g_grid_x[coords[0]+1]) &
		(g_comp[:, 1] >= g_grid_y[coords[1]]) &
		(g_comp[:, 1] <= g_grid_y[coords[1]+1])
	)

	mask_idx = np.argwhere(mask == True)

	if len(g_comp[mask, 2]) > 2:

		noise_cluster_ratio = 0.2
		
		model = DBSCAN(
			eps=3.0*g_res,
			min_samples=10,
			metric='euclidean',
			algorithm='kd_tree',
			leaf_size=30,
			n_jobs=1
		)
		model.fit(g_comp[mask, :])

		labels, counts = np.unique(model.labels_, return_counts=True)
		noise_idx = np.array([np.argwhere(model.labels_ == l).ravel() for l,c in zip(labels, counts) if (c/len(model.labels_)) < noise_cluster_ratio])
		noise_idx = list(itertools.chain.from_iterable(noise_idx))
		noise_idx.sort()

		return mask_idx[noise_idx].ravel().tolist()

		'''
		pca = IncrementalPCA(n_components=3, batch_size=1000)
		pca.fit(g_comp[mask, :])
		comp_local = pca.transform(g_comp[mask, :])
		diff = comp_local[:, 2]

		gap_threshold = 3.0 # Look for gaps > 3.0 mm
		sorted_idx = np.argsort(diff)
		d_diff = np.diff(diff[sorted_idx])
		d_indices = np.arange(d_diff.shape[0])
		d_splitted = np.split(d_indices, np.argwhere(d_diff > gap_threshold).ravel() + 1)

		if len(d_splitted) > 1:
			
			noise_idx = [d_sublist.tolist() for d_sublist in d_splitted if (d_sublist.shape[0]/d_indices.shape[0]) < 0.1] # Keep groups > 10%
			
			if len(noise_idx) > 0:
				noise_idx = list(itertools.chain.from_iterable(noise_idx))
				return mask_idx[sorted_idx[noise_idx]].ravel().tolist()
			else:
				return []
		
		else:
			return []
		'''
	else:
		return []


def noise_removal(pcd, resolution, visualize=False):
	
	points = np.asarray(pcd.points)
	normals = np.asarray(pcd.normals)

	pca = IncrementalPCA(n_components=3, batch_size=1000)
	pca.fit(points)
	comp = pca.transform(points)

	x_min =  comp[:, 0].min()
	x_max =  comp[:, 0].max()
	y_min =  comp[:, 1].min()
	y_max =  comp[:, 1].max()

	offset = 0.01
	x_offset = offset * np.abs(x_max - x_min)
	y_offset = offset * np.abs(y_max - y_min)

	grid_size = 300.0 # 30 x 30 cm subgrid
	x_resolution = int(np.abs((x_max+x_offset) - (x_min-x_offset)) / (grid_size / resolution)) * 1j
	y_resolution = int(np.abs((y_max+y_offset) - (y_min-y_offset)) / (grid_size / resolution)) * 1j

	grid_x = np.mgrid[(x_min-x_offset):(x_max+x_offset):x_resolution]
	grid_y = np.mgrid[(y_min-y_offset):(y_max+y_offset):y_resolution]

	grid_coords = np.array(list(itertools.product(range(0, grid_x.shape[0]-1), range(0, grid_y.shape[0]-1))))

	print('Noise Detection: Local Density-based Spatial Clustering')
	cpu_count = mp.cpu_count()
	pool = mp.Pool(
		processes=cpu_count,
		initializer=check_noise_init,
		initargs=(comp, grid_x, grid_y, resolution)
	)

	g_density_noise_indices = pool.map(check_noise_worker, grid_coords)
	g_density_noise_indices = list(itertools.chain.from_iterable(g_density_noise_indices))
	g_density_non_noise_indices = np.delete(np.arange(points.shape[0]), g_density_noise_indices)
	pool.close()
	pool.join()

	pcd_noise_g_density = pcd.select_down_sample(g_density_noise_indices)
	pcd_inlier = pcd.select_down_sample(g_density_noise_indices, invert=True)

	print('Noise Detection: Global Density-based Spatial Clustering')
	noise_cluster_ratio = 0.05
	
	model = DBSCAN(
		eps=3.0 * resolution,
		min_samples=10,
		metric='euclidean',
		algorithm='ball_tree',
		leaf_size=30,
		n_jobs=-1
	)

	'''
	model = OPTICS(
		min_samples=10,
		metric='euclidean',
		algorithm='ball_tree',
		leaf_size=30,
		n_jobs=-1
	)
	'''
	model.fit(np.asarray(pcd_inlier.points))
	labels, counts = np.unique(model.labels_, return_counts=True)

	l_density_noise_indices = np.array([np.argwhere(model.labels_ == l).ravel() for l,c in zip(labels, counts) if (c/len(model.labels_)) < noise_cluster_ratio])
	l_density_noise_indices = list(itertools.chain.from_iterable(l_density_noise_indices))
	l_density_noise_indices.sort()

	pcd_noise_l_density = pcd_inlier.select_down_sample(l_density_noise_indices)
	pcd_inlier = pcd_inlier.select_down_sample(l_density_noise_indices, invert=True)
	
	print('Noise Detection: Distance Deviation')
	cl, distance_non_noise_indices = pcd_inlier.remove_statistical_outlier(
		nb_neighbors=50,
		std_ratio=3.0
	)

	pcd_noise_distance = pcd_inlier.select_down_sample(distance_non_noise_indices, invert=True)
	pcd_inlier = pcd_inlier.select_down_sample(distance_non_noise_indices)

	pcd_inlier.paint_uniform_color([0.7, 0.7, 0.7])
	pcd_noise_g_density.paint_uniform_color([1.0, 0.19, 0.15]) # Red
	pcd_noise_l_density.paint_uniform_color([0.27, 0.46, 1.0]) # Blue
	pcd_noise_distance.paint_uniform_color([0.1, 1.0, 0.31]) # Green


	if visualize:
		visualize_point_clouds([
			pcd_inlier, 
			pcd_noise_g_density,
			pcd_noise_l_density,
			pcd_noise_distance
		])


	return pcd_inlier

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

def transform_point_cloud(pcd, visualize=False):

	points = np.asarray(pcd.points)
	normals = np.asarray(pcd.normals)

	pca = IncrementalPCA(n_components=3, batch_size=1000)
	pca.fit(points)
	components = pca.components_
	mean = pca.mean_
	normals_mean = np.dot(normals.mean(axis=0), components.T)

	if np.dot(components[2], np.cross(components[0], components[1])) < 0.0:
		components[2] *= -1.0

	if np.dot(components[2], normals.mean(axis=0)) < 0.0:
		components[0] *= -1.0
		components[2] *= -1.0

	points_transformed = np.dot(points - mean, components.T)
	normals_transformed = np.dot(normals, components.T)

	degrees = []
	ratios = []
	for d in range(0, 180, 1):
		x, y = rotate(
			points_transformed[:, 0],
			points_transformed[:, 1],
			np.radians(d)
		)

		x_diff = x.max() - x.min()
		y_diff = y.max() - y.min()
		degrees.append(d)
		ratios.append(x_diff/y_diff)

	idx = np.argsort(ratios)
	rot_z_cc_deg = degrees[idx[-1]]

	points_transformed[:, 0], points_transformed[:, 1] = rotate(
		points_transformed[:, 0],
		points_transformed[:, 1],
		np.radians(rot_z_cc_deg)
	)

	normals_transformed[:, 0], normals_transformed[:, 1] = rotate(
		normals_transformed[:, 0],
		normals_transformed[:, 1],
		np.radians(rot_z_cc_deg)
	)

	wrong_direction_mask = normals_transformed[:, 2] < 0.0
	normals_transformed[wrong_direction_mask, :] = -1.0 * normals_transformed[wrong_direction_mask, :]

	scan_stats = {
		'width': (points_transformed[:, 0].max() - points_transformed[:, 0].min()) / (100*10),
		'height': (points_transformed[:, 1].max() - points_transformed[:, 1].min()) / (100*10),
		'depth': (points_transformed[:, 2].max() - points_transformed[:, 2].min()) / (100*10),
		'rot_z_cc_deg': rot_z_cc_deg,
		'n_wrong_n_dir': np.sum(wrong_direction_mask)
	}

	print('Rotating around Z-axis ' + str(rot_z_cc_deg) + ' degrees.')
	print('Number of Wrong Normal Directions: ' + str(scan_stats['n_wrong_n_dir']))
	print('Scan Width: ' + ('%.2f' % scan_stats['width']) + ' m')
	print('Scan Height: ' + ('%.2f' % scan_stats['height']) + ' m')
	print('Scan Depth: ' + ('%.2f' % scan_stats['depth']) + ' m')

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points_transformed)
	pcd.normals = o3d.utility.Vector3dVector(normals_transformed)

	#c = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0, origin=np.array([0., 0., 0.]))

	if visualize:

		visualize_point_clouds([pcd])

	return pcd, scan_stats, components, mean, rot_z_cc_deg