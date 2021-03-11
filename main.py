
import multiprocessing
multiprocessing.freeze_support()

import pickle
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree._utils
import cython
import sklearn
import sklearn.utils._cython_blas
import joblib

import os
import sys
import time
import json
import shutil
import psutil
import gc
import traceback

import numpy as np
import pandas as pd
import open3d as o3d

from tqdm import tqdm
from pc_mesh import *
from pc_image import *

# Strange coordinate system
#"Ingot_high_high",
#"Tanum_184_1_hfg_fig_1"

# High Res
#"Tanum_148_1",
#"Hemsta_hällen",
#"Askum_44_hires_clean",
#"Kville_165_boat_2_high_res",
#"Bottna_110_svärd_1_high_res",
#"Brastad_617_yta_B_hfg_spear_high_res"

#"Tanum_248_1_lurblowers",
#"Tanum_151_1_part_2",
#"Tanum_89_1_merge",
#"Tanum_405_1_lureblowers",
#"Tanum_89_1_hfg_8",
#"Tanum_2365_Arendal_ny_fig_1",
#"Tanum_283_1",
#"Askum_67_1_komplett",
#"Tanum_406_1_top_part"

PROCESS = psutil.Process(os.getpid())
GIGA = 10 ** 9

def print_memory_usage():

	mem = psutil.virtual_memory()

	total = mem.total / GIGA
	available = mem.available / GIGA
	free = mem.free / GIGA
	used = mem.used / GIGA
	percent = mem.percent

	proc = PROCESS.memory_info()[1] / GIGA

	print('Memory Info:')
	print('\tTotal: ' + ('%.2f' % total) + ' GB')
	print('\tSystem Used: ' + ('%.2f' % used) + ' GB')
	print('\tProcess Used: ' + ('%.2f' % proc) + ' GB')
	print('\tPercentage: ' + ('%.2f' % percent) + ' %')
	print('\tFree: ' + ('%.2f' % free) + ' GB')
	print('\tAvailable: ' + ('%.2f' % available) + ' GB')
	print('')

def log_exception(exception: BaseException, expected: bool = True):

	output = "[{}] {}: {}".format('EXPECTED' if expected else 'UNEXPECTED', type(exception).__name__, exception)
	print(output)

	exc_type, exc_value, exc_traceback = sys.exc_info()
	traceback.print_tb(exc_traceback)

def read_config():

	with open('config.json', 'r') as file:
		return json.load(file)

def ms_output(seconds):
	
	return str(pd.to_timedelta(seconds, unit='s'))

def create_panel_folder(path):

	if not os.path.exists(path):
		os.makedirs(path)
		os.makedirs(path + '/img')
		os.makedirs(path + '/img/topo_maps')
		os.makedirs(path + '/img/enhanced_topo_maps')
		os.makedirs(path + '/img/blended_maps')
		os.makedirs(path + '/arrays')


def main():

	print('\n\n\n')

	total_time_start = time.time()
	config = read_config()

	files = []
	for dp, dn, filenames in os.walk(config['data_path']):
		for file in filenames:
			if os.path.splitext(file)[1] in ['.stl', '.ply']:
				files.append(('_'.join(file.split('.')[0].split(' ')), os.path.join(dp, file), '_'.join(dp.replace(config['data_path'], config['save_path']).split(' '))))

	
	if len(config['process_panels']) > 0:
		files = [(sid, read_path, save_path) for sid, read_path, save_path in files if sid in config['process_panels']]

	print('Nr of Scans to Process: ' + str(len(files)))

	meta_data = {
		'id': [],
		'mesh_edge_resolution': [],
		'coord_units_updated': [],
		'mesh_n_vertices': [],
		'pc_n_points': [],
		'pc_cleaned_n_points': [],
		'n_wrong_n_dir': [],
		'rot_z_cc_deg': [],
		'scan_width': [],
		'scan_height': [],
		'scan_depth': [],
		'scan_depth_per_m2': [],
		'pc_resolution': [],
		'img_width': [],
		'img_height': [],
		'proc_time': []
	}

	scan_counter = 0
	for scan_id, read_path, save_path in files[:]:

		scan_counter += 1

		print('Processing: ' + scan_id + ' (' + str(scan_counter) + '/' + str(len(files)) + ')')
		print_memory_usage()
		print('')
		scan_time_start = time.time()

		# Create Data Folder
		########################################################

		create_panel_folder(os.path.join(save_path, scan_id))
		
		meta_data['id'].append(scan_id)
		
		########################################################

		# Read Data File
		# Compute Normals
		# Convert to Point Cloud
		########################################################

		print('Creating Point Cloud.')
		time_start = time.time()

		mesh, mesh_edge_resolution = read_mesh(
			read_path,
			down_sample=config['down_sample_mesh'],
			visualize=config['visualize_steps'],
		)

		meta_data['mesh_edge_resolution'].append(mesh_edge_resolution)
		meta_data['mesh_n_vertices'].append(len(mesh.vertices))
		#o3d.write_triangle_mesh(os.path.join(save_path, scan_id) + '/' +'original.stl', mesh)

		pcd, mesh_edge_resolution = create_point_cloud(
			mesh,
			mesh_edge_resolution,
			update_resolution=config['update_resolution'],
			visualize=config['visualize_steps']
		)

		if np.abs(meta_data['mesh_edge_resolution'][-1] - mesh_edge_resolution) < 0.001:
			meta_data['coord_units_updated'].append(False)
		else:
			meta_data['coord_units_updated'].append(True)

		meta_data['pc_n_points'].append(len(pcd.points))

		o3d.io.write_point_cloud(os.path.join(save_path, scan_id) + '/' +'original.pcd', pcd)

		time_end = time.time()
		print('Execution Time: ' + ms_output(time_end - time_start))
		print('')

		########################################################

		# Outlier Removal
		########################################################

		print('Removing Noise.')
		time_start = time.time()

		pcd = noise_removal(
			pcd,
			mesh_edge_resolution,
			visualize=config['visualize_steps']
		)

		meta_data['pc_cleaned_n_points'].append(len(pcd.points))

		#o3d.io.write_point_cloud(os.path.join(save_path, scan_id) + '/' +'cleaned.pcd' , pcd)

		time_end = time.time()
		print('Execution Time: ' + ms_output(time_end - time_start))
		print('')
		
		########################################################

		# Transform Point Cloud through PCA
		########################################################

		print('Transforming Point Cloud.')
		time_start = time.time()

		pcd, scan_stats, components, mean, rotation = transform_point_cloud(
			pcd,
			visualize=config['visualize_steps']
		)

		meta_data['n_wrong_n_dir'].append(scan_stats['n_wrong_n_dir'])
		meta_data['rot_z_cc_deg'].append(scan_stats['rot_z_cc_deg'])
		meta_data['scan_width'].append(scan_stats['width'])
		meta_data['scan_height'].append(scan_stats['height'])
		meta_data['scan_depth'].append(scan_stats['depth'])
		meta_data['scan_depth_per_m2'].append(scan_stats['depth'] / (scan_stats['height'] * scan_stats['width']))

		o3d.io.write_point_cloud(os.path.join(save_path, scan_id) + '/' +'transformed.pcd', pcd)
		np.savez_compressed(
			os.path.join(save_path, scan_id) + '/arrays/transform_meta.npz',
			components=components,
			mean=mean,
			rotation=rotation
		)

		time_end = time.time()
		print('Execution Time: ' + ms_output(time_end - time_start))
		print('')
		
		########################################################

		# Create Images from Point Cloud
		########################################################
		
		print('Generating Images.')
		time_start = time.time()

		depth_map, normal_map, pc_resolution, pix_to_coords_map = generate_images(
			pcd,
			visualize=config['visualize_steps']
		)

		meta_data['pc_resolution'].append(pc_resolution)
		meta_data['img_width'].append(depth_map.shape[1])
		meta_data['img_height'].append(depth_map.shape[0])

		np.savez_compressed(os.path.join(save_path, scan_id) + '/arrays/pix_to_coords_map.npz', data=pix_to_coords_map)
		np.savez_compressed(os.path.join(save_path, scan_id) + '/arrays/depth_map.npz', data=depth_map)
		np.savez_compressed(os.path.join(save_path, scan_id) + '/arrays/normal_map.npz', data=normal_map)

		time_end = time.time()
		print('Execution Time: ' + ms_output(time_end - time_start))
		print('')
		
		########################################################

		# Create Topography Map
		########################################################

		print('Creating Topography Maps.')
		time_start = time.time()


		topo_map_texture_level = create_topo_map(
			depth_map,
			sigma=(4.0 / pc_resolution),
			visualize=config['visualize_steps']
		)

		topo_map_object_level = create_topo_map(
			depth_map,
			sigma=(32.0 / pc_resolution),
			visualize=config['visualize_steps']
		)

		np.savez_compressed(os.path.join(save_path, scan_id) + '/arrays/topo_map_texture_level.npz', data=topo_map_texture_level)
		np.savez_compressed(os.path.join(save_path, scan_id) + '/arrays/topo_map_object_level.npz', data=topo_map_object_level)

		time_end = time.time()
		print('Execution Time: ' + ms_output(time_end - time_start))
		print('')
		
		########################################################

		# Enhance Topo Map
		########################################################
		
		print('Enhancing Topo Maps.')
		time_start = time.time()

		enhanced_topo_map_texture_level = enhance_topo_map(
			topo_map_texture_level,
			visualize=config['visualize_steps']
		)

		mask = clean_mask(
			~np.isnan(enhanced_topo_map_texture_level),
			visualize=config['visualize_steps']
		)

		enhanced_topo_map_object_level = enhance_topo_map(
			topo_map_object_level,
			visualize=config['visualize_steps']
		)

		enhanced_topo_map_texture_level[~mask] = np.nan
		enhanced_topo_map_object_level[~mask] = np.nan

		np.savez_compressed(os.path.join(save_path, scan_id) + '/arrays/enhanced_topo_map_texture_level.npz', data=enhanced_topo_map_texture_level)
		np.savez_compressed(os.path.join(save_path, scan_id) + '/arrays/enhanced_topo_map_object_level.npz', data=enhanced_topo_map_object_level)

		time_end = time.time()
		print('Execution Time: ' + ms_output(time_end - time_start))
		print('')
		
		########################################################

		# Save Images.
		########################################################
		
		print('Saving Images.')
		time_start = time.time()

		save_images(
			os.path.join(save_path, scan_id) + '/img/',
			depth_map,
			normal_map,
			{
				'texture_level': topo_map_texture_level,
				'object_level': topo_map_object_level
			},
			{
				'texture_level': enhanced_topo_map_texture_level,
				'object_level': enhanced_topo_map_object_level
			},
		)

		time_end = time.time()
		print('Execution Time: ' + ms_output(time_end - time_start))
		print('')
		
		########################################################

		del depth_map
		del normal_map
		del pix_to_coords_map
		del	topo_map_texture_level
		del topo_map_object_level
		del enhanced_topo_map_texture_level
		del enhanced_topo_map_object_level
		gc.collect()

		scan_time_end = time.time()
		meta_data['proc_time'].append(pd.to_timedelta(scan_time_end - scan_time_start, unit='s'))
		print('Scan Execution Time: ' + ms_output(scan_time_end - scan_time_start))
		print_memory_usage()
		print('\n########################################################')
		print('\n\n\n')

	df_meta_data = pd.DataFrame(meta_data)
	df_meta_data.to_csv(config['meta_data_path'], index=False)

	total_time_end = time.time()
	print('Processing finished.')
	print('Total Execution Time: ' + ms_output(total_time_end - total_time_start))
	print('\n########################################################')
	print('\n\n\n')


if __name__ == '__main__':

	try:
		main()
	except MemoryError as error:
		log_exception(error)
	except Exception as exception:
		log_exception(exception, False)

