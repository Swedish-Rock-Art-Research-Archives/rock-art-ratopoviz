
import skimage
import itertools
import warnings

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.patches as mpatches
import multiprocessing as mp
import scipy as sp

from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import PowerTransformer
from itertools import repeat
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import disk

if __name__ == '__main__':
	mp.freeze_support()

GRAY_COLORMAP = 'gist_yarg'
TOPO_MAP_COLORMAP = 'jet'
DEPTH_MAP_COLORMAP = 'Spectral'
BLENDED_MAP_COLORMAP = 'jet'
MASK_COLORMAP = 'gray'

def clean_mask(mask, visualize=False):

	print('Binary Opening.')

	mask_opened = skimage.morphology.binary_opening(
		mask,
		disk(10)
	)

	print('Removing Small Blobs.')

	mask_cleaned = skimage.morphology.remove_small_objects(
		mask_opened,
		min_size=int(0.05 * np.sum(mask_opened)),
		connectivity=1
	)

	if visualize:

		fig = plt.figure(figsize=(16,16))

		ax = fig.add_subplot(2, 2, 1)
		ax.grid(False)
		cm = plt.get_cmap(MASK_COLORMAP) 
		plt.imshow(mask, cmap=cm, origin='upper')
		plt.title('Original Mask')

		ax = fig.add_subplot(2, 2, 2)
		ax.grid(False)
		cm = plt.get_cmap(MASK_COLORMAP) 
		plt.imshow(mask_opened, cmap=cm, origin='upper')
		plt.title('Mask Opened')

		ax = fig.add_subplot(2, 2, 3)
		ax.grid(False)
		cm = plt.get_cmap(MASK_COLORMAP) 
		plt.imshow(mask_cleaned, cmap=cm, origin='upper')
		plt.title('Mask Cleaned')

		plt.show()

	return mask_cleaned

def enhance_topo_map(topo_map, visualize=False):

	print('Removing Extreme Values.')

	q1, q3 = np.percentile(topo_map[~np.isnan(topo_map)].flatten(), [25,75])
	iqr = q3 - q1
	lower_bound = q1 - (iqr * 6.0)
	upper_bound = q3 + (iqr * 10.0)
	
	enhanced_topo_map_cleaned = topo_map.copy()

	mask = ~np.isnan(enhanced_topo_map_cleaned)
	mask[mask] &= enhanced_topo_map_cleaned[mask] < upper_bound
	mask[mask] &= enhanced_topo_map_cleaned[mask] > lower_bound
	enhanced_topo_map_cleaned[~mask] = np.nan

	print('Removing Small Blobs.')

	mask = ~np.isnan(enhanced_topo_map_cleaned)
	mask_cleaned = skimage.morphology.remove_small_objects(
		mask,
		min_size=int(0.05 * np.sum(mask)),
		connectivity=1
	)

	enhanced_topo_map_cleaned[~mask_cleaned] = np.nan
	
	print('Log Scaling.')

	enhanced_topo_map_scaled = enhanced_topo_map_cleaned.copy()
	
	mask = ~np.isnan(enhanced_topo_map_scaled)
	mask[mask] &= enhanced_topo_map_scaled[mask] >= 0.0
	enhanced_topo_map_scaled[mask] = np.log(enhanced_topo_map_scaled[mask] + 1.0)

	mask = ~np.isnan(enhanced_topo_map_scaled)
	mask[mask] &= enhanced_topo_map_scaled[mask] < 0.0
	enhanced_topo_map_scaled[mask] = -1.0 * np.log((-1.0 * enhanced_topo_map_scaled[mask]) + 1.0)
	
	print('Adaptive Histogram Equalization.')

	enhanced_topo_map_equalized = enhanced_topo_map_scaled.copy()

	_min = np.nanmin(enhanced_topo_map_equalized)
	_max = np.nanmax(enhanced_topo_map_equalized)

	nan_mask = np.isnan(enhanced_topo_map_equalized)
	enhanced_topo_map_equalized = (enhanced_topo_map_equalized - _min) / (_max - _min)
	enhanced_topo_map_equalized[nan_mask] = 0.0

	with warnings.catch_warnings():

		warnings.simplefilter("ignore")

		enhanced_topo_map_equalized = skimage.img_as_uint(enhanced_topo_map_equalized)
		enhanced_topo_map_equalized = skimage.exposure.equalize_adapthist(
			enhanced_topo_map_equalized,
			kernel_size=1000,
			clip_limit=0.01
		)
		enhanced_topo_map_equalized[nan_mask] = np.nan

	if visualize:

		fig = plt.figure(figsize=(16,16))

		ax = fig.add_subplot(2, 2, 1)
		ax.grid(False)
		cm = plt.get_cmap(TOPO_MAP_COLORMAP) 
		cm.set_bad(color='grey')
		img = plt.imshow(topo_map, cmap=cm, origin='upper')
		plt.colorbar(img)
		plt.title('0. Topo Map')

		ax = fig.add_subplot(2, 2, 2)
		ax.grid(False)
		cm = plt.get_cmap(TOPO_MAP_COLORMAP) 
		cm.set_bad(color='grey')
		img = plt.imshow(enhanced_topo_map_cleaned, cmap=cm, origin='upper')
		plt.colorbar(img)
		plt.title('1. Cleaned Topo Map')

		ax = fig.add_subplot(2, 2, 3)
		ax.grid(False)
		cm = plt.get_cmap(TOPO_MAP_COLORMAP) 
		cm.set_bad(color='grey')
		img = plt.imshow(enhanced_topo_map_scaled, cmap=cm, origin='upper')
		plt.colorbar(img)
		plt.title('2. Log Scaled Topo Map')

		ax = fig.add_subplot(2, 2, 4)
		ax.grid(False)
		cm = plt.get_cmap(TOPO_MAP_COLORMAP) 
		cm.set_bad(color='grey')
		img = plt.imshow(enhanced_topo_map_equalized, cmap=cm, origin='upper')
		plt.colorbar(img)
		plt.title('3. Equalized Topo Map')
	
		plt.show()

	return enhanced_topo_map_equalized

def flatten_normal_map(normal_map, visualize=False):

	sigma = 64.0

	normal_map_mask = np.isnan(normal_map)
	normal_map_smoothed = normal_map.copy()
	normal_map_smoothed[normal_map_mask] = 0.0
	normal_map_smoothed = skimage.filters.gaussian(normal_map_smoothed, sigma=sigma, multichannel=True)
	normal_map_smoothed_nans = skimage.filters.gaussian((~normal_map_mask).astype(np.float), sigma=sigma, multichannel=True)
	
	normal_map_smoothed = np.true_divide(
		normal_map_smoothed,
		normal_map_smoothed_nans,
		out=np.zeros_like(normal_map_smoothed),
		where=normal_map_smoothed_nans != 0.0
	)

	norms = np.zeros((normal_map_smoothed.shape[0], normal_map_smoothed.shape[1], 3))
	norms[:,:,0] = np.linalg.norm(normal_map_smoothed, axis=2)
	norms[:,:,1] = norms[:,:,0]
	norms[:,:,2] = norms[:,:,0]
	normal_map_smoothed = np.true_divide(
		normal_map_smoothed,
		norms,
		out=np.full_like(normal_map_smoothed, np.nan),
		where=norms != 0.0
	)

	flat_normal_map = normal_map + (np.array([0.0, 0.0, 1.0]) - normal_map_smoothed)
	norms = np.zeros((flat_normal_map.shape[0], flat_normal_map.shape[1], 3))
	norms[:,:,0] = np.linalg.norm(flat_normal_map, axis=2)
	norms[:,:,1] = norms[:,:,0]
	norms[:,:,2] = norms[:,:,0]
	flat_normal_map = np.true_divide(
		flat_normal_map,
		norms,
		out=np.zeros_like(flat_normal_map),
		where=norms != 0.0
	)

	nm_min = -1.0
	nm_max = 1.0
	normal_map_scaled = (normal_map - nm_min) / (nm_max - nm_min)
	normal_map_smoothed_scaled = (normal_map_smoothed - nm_min) / (nm_max - nm_min)
	flat_normal_map_scaled = (flat_normal_map - nm_min) / (nm_max - nm_min)

	if visualize:

		fig = plt.figure(figsize=(16,16))

		ax = fig.add_subplot(2, 2, 1)
		ax.grid(False)
		plt.imshow(normal_map_scaled, origin='upper')
		plt.title('Normal Map')

		ax = fig.add_subplot(2, 2, 2)
		ax.grid(False)
		plt.imshow(normal_map_smoothed_scaled, origin='upper')
		plt.title('Normal Map Smoothed')

		ax = fig.add_subplot(2, 2, 3)
		ax.grid(False)
		plt.imshow(flat_normal_map_scaled, origin='upper')
		plt.title('Flat Normal Map')
	
		plt.show()

	return flat_normal_map

def create_topo_map(depth_map, sigma, visualize=False):

	depth_map_mask = np.isnan(depth_map)
	depth_map_smoothed = depth_map.copy()
	depth_map_smoothed[depth_map_mask] = 0.0
	depth_map_smoothed = skimage.filters.gaussian(depth_map_smoothed, sigma=sigma)
	depth_map_smoothed_nans = skimage.filters.gaussian((~depth_map_mask).astype(np.float), sigma=sigma)
	depth_map_smoothed = np.true_divide(
		depth_map_smoothed,
		depth_map_smoothed_nans,
		out=np.zeros_like(depth_map_smoothed),
		where=depth_map_smoothed_nans != 0.0
	)
	topo_map = depth_map - depth_map_smoothed
	
	if visualize:

		fig = plt.figure(figsize=(16,16))

		ax = fig.add_subplot(2, 2, 1)
		ax.grid(False)
		cm = plt.get_cmap(DEPTH_MAP_COLORMAP) 
		cm.set_bad(color='grey')
		img = plt.imshow(depth_map, cmap=cm, origin='upper')
		plt.colorbar(img)
		plt.title('Depth Map')

		ax = fig.add_subplot(2, 2, 2)
		ax.grid(False)
		cm = plt.get_cmap(DEPTH_MAP_COLORMAP) 
		cm.set_bad(color='grey')
		img = plt.imshow(depth_map_smoothed, cmap=cm, origin='upper')
		plt.colorbar(img)
		plt.title('Depth Map Smoothed - Sigma: ' + ('%.2f' % sigma))

		ax = fig.add_subplot(2, 2, 3)
		ax.grid(False)
		cm = plt.get_cmap(TOPO_MAP_COLORMAP) 
		cm.set_bad(color='grey')
		img = plt.imshow(topo_map, cmap=cm, origin='upper')
		plt.colorbar(img)
		plt.title('Topography Map - Sigma: ' + ('%.2f' % sigma))
	
		plt.show()

	return topo_map

def generate_images(pcd, visualize=False):

	points = np.asarray(pcd.points)
	normals = np.asarray(pcd.normals)

	print('Creating KDTree.')
	kd_tree = sp.spatial.cKDTree(np.c_[points[:, 0], points[:, 1]])
	dist,_ = kd_tree.query(np.c_[points[:, 0], points[:, 1]], k=range(2,6))
	dist_nn = np.array([d.mean() for d in dist])
	pc_resolution = dist_nn.mean()

	print('Point Cloud Resolution: ' + ('%.2f' % pc_resolution) + ' mm')

	x_max = points[:, 0].max()
	x_min = points[:, 0].min()
	y_max = points[:, 1].max()
	y_min = points[:, 1].min()

	offset = 0.01
	x_offset = offset * np.abs(x_max - x_min)
	y_offset = offset * np.abs(y_max - y_min)

	grid_x, grid_y = np.mgrid[
		(x_min-x_offset):(x_max+x_offset):(pc_resolution/2.0),
		(y_min-y_offset):(y_max+y_offset):(pc_resolution/2.0)
	]

	grid_x = np.transpose(grid_x)
	grid_x = np.flip(grid_x, axis=0)
	grid_y = np.transpose(grid_y)
	grid_y = np.flip(grid_y, axis=0)

	print('Image Size: ' + str(grid_x.shape[1]) + ' x ' + str(grid_x.shape[0]))
	
	grid_x_flattened = grid_x.flatten()
	grid_y_flattened = grid_y.flatten()

	print('Creating Depth Map.')

	depth_map = sp.interpolate.griddata(
		(
			points[:, 0],
			points[:, 1]
		),
		points[:, 2],
		(
			grid_x,
			grid_y
		),
		method='linear',
		rescale=False
	)

	print('Removing Convex Shape.')
	dist, _ = kd_tree.query(np.c_[grid_x_flattened, grid_y_flattened], k=1)
	dist = dist.reshape(grid_x.shape)
	depth_map[dist > pc_resolution + 3.0*dist_nn.std()] = np.nan

	if visualize:

		fig = plt.figure(figsize=(20,10))

		img_sub_perc = [0.002, 0.001]
		x_mean = points[:, 0].mean()
		y_mean = points[:, 1].mean()

		for i in range(len(img_sub_perc)):

			x_diff = img_sub_perc[i] * np.abs(x_max - x_min)
			y_diff = img_sub_perc[i] * np.abs(y_max - y_min)

			pc_indices, = np.where(
				np.logical_and(
					(points[:, 0] < x_mean + x_diff) & (points[:, 0] > x_mean - x_diff),
					(points[:, 1] < y_mean + y_diff) & (points[:, 1] > y_mean - y_diff)
				)
			)

			grid_indices, = np.where(
				np.logical_and(
					(grid_x_flattened < x_mean + x_diff) & (grid_x_flattened > x_mean - x_diff),
					(grid_y_flattened < y_mean + y_diff) & (grid_y_flattened > y_mean - y_diff)
				)
			)

			ax = fig.add_subplot(1,2,i+1)
			ax.grid(False)
			ax.set_facecolor('black')
			ax.set_xlim(x_mean - x_diff, x_mean + x_diff)
			ax.set_ylim(y_mean - y_diff, y_mean + y_diff)

			ax.scatter(
				x=grid_x_flattened[grid_indices],
				y=grid_y_flattened[grid_indices],
				c='#e41a1c',
				marker='s',
				alpha=1.0
			)

			# Plot original point cloud data.
			ax.scatter(
				x=points[pc_indices, 0],
				y=points[pc_indices, 1],
				c='#ffff33',
				alpha=1.0
			)

			pc_patch = mpatches.Patch(color='#ffff33', label='Point Cloud')
			grid_patch = mpatches.Patch(color='#e41a1c', label='Pixels')
			plt.legend(handles=[pc_patch, grid_patch], loc=1, fontsize=12)

			plt.xlabel('X', fontsize=12)
			plt.ylabel('Y', fontsize=12)
			plt.title('Resolution Comparison (Zoom Level: ' + str(img_sub_perc[i]) + ')')

		plt.show()

		# Add figure showing resulting image.
		fig = plt.figure(figsize=(16,12))
		ax = plt.gca()
		ax.grid(False)
		cm = plt.get_cmap(DEPTH_MAP_COLORMAP) 
		cm.set_bad(color='grey')
		img = plt.imshow(depth_map, cmap=cm, origin='upper')
		plt.colorbar(img)
		plt.title('Depth Map')
		plt.show()

	print('Creating Normal Map.')

	normal_map = sp.interpolate.griddata(
		(
			points[:, 0],
			points[:, 1]
		),
		normals[:, :],
		(
			grid_x,
			grid_y
		),
		method='linear',
		rescale=False
	)

	normal_map[np.isnan(depth_map)] = np.nan

	if visualize:

		nm_min = -1.0
		nm_max = 1.0
		normal_map_scaled = (normal_map - nm_min) / (nm_max - nm_min)

		fig = plt.figure(figsize=(16,12))
		ax = plt.gca()
		ax.grid(False)
		plt.imshow(normal_map_scaled, origin='upper')
		plt.title('Normal Map')
		plt.show()

	return depth_map, normal_map, pc_resolution, np.array([grid_x, grid_y])

def save_images(save_path, depth_map, normal_map, topo_maps, enhanced_topo_maps):

	print('Saving Depth Map.')
	with warnings.catch_warnings():

		warnings.simplefilter('ignore')

		cmap = plt.cm.get_cmap(DEPTH_MAP_COLORMAP)
		cnorm = colors.Normalize(vmin=np.nanmin(depth_map), vmax=np.nanmax(depth_map))
		smap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
		nan_mask = np.isnan(depth_map)

		depth_map_img = depth_map.copy()
		depth_map_img[nan_mask] = 0.0
		depth_map_img = skimage.img_as_ubyte(smap.to_rgba(depth_map_img)[:,:,:3])
		depth_map_img[nan_mask] = [0, 0, 0]

		skimage.io.imsave(save_path + 'depth_map.png', depth_map_img)

		fig = plt.figure(figsize=(8,1))
		ax = plt.gca()
		cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=cnorm, orientation='horizontal')
		cb.set_label('Depth Colorbar')
		plt.tight_layout()
		plt.savefig(save_path + 'depth_map_colorbar.png')
		plt.close()

	print('Saving Normal and Derivative Map.')
	with warnings.catch_warnings():

		warnings.simplefilter('ignore')

		nm_min = -1.0
		nm_max = 1.0
		nan_mask = np.isnan(normal_map[:,:,0])

		normal_map_img = (normal_map - nm_min) / (nm_max - nm_min)
		normal_map_img[nan_mask] = [0.0, 0.0, 0.0]
		normal_map_img = skimage.img_as_ubyte(normal_map_img)

		derivative_map_img = normal_map_img.copy()
		derivative_map_img[:,:,2] = 0

		skimage.io.imsave(save_path + 'normal_map.png', normal_map_img)
		skimage.io.imsave(save_path + 'derivative_map.png', derivative_map_img)


	print('Saving Topo Maps.')
	with warnings.catch_warnings():

		warnings.simplefilter('ignore')

		for key, topo_map in topo_maps.items():

			cmap = plt.cm.get_cmap(TOPO_MAP_COLORMAP)
			cnorm = colors.Normalize(vmin=np.nanmin(topo_map), vmax=np.nanmax(topo_map))
			smap = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
			nan_mask = np.isnan(topo_map)

			topo_map_img = topo_map.copy()
			topo_map_img[nan_mask] = 0.0
			topo_map_img = skimage.img_as_ubyte(1.0 - cnorm(topo_map_img))
			topo_map_img[nan_mask] = 0

			skimage.io.imsave(save_path + 'topo_maps/topo_map_' + key + '_grey.png', topo_map_img)
			
			topo_map_img = topo_map.copy()
			topo_map_img[nan_mask] = 0.0
			topo_map_img = skimage.img_as_ubyte(smap.to_rgba(topo_map_img)[:,:,:3])
			topo_map_img[nan_mask] = [0, 0, 0]

			skimage.io.imsave(save_path + 'topo_maps/topo_map_' + key + '.png', topo_map_img)

			fig = plt.figure(figsize=(8,1))
			ax = plt.gca()
			cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=cnorm, orientation='horizontal')
			cb.set_label('Topo Colorbar')
			plt.tight_layout()
			plt.savefig(save_path + 'topo_maps/topo_map_ ' + key + ' _colorbar.png')
			plt.close()

	print('Saving Enhanced Topo Maps.')
	with warnings.catch_warnings():

		warnings.simplefilter('ignore')

		for key, enhanced_topo_map in enhanced_topo_maps.items():

			cmap = plt.cm.get_cmap(TOPO_MAP_COLORMAP)
			smap = plt.cm.ScalarMappable(norm=None, cmap=cmap)
			nan_mask = np.isnan(enhanced_topo_map)

			enhanced_topo_map_img = enhanced_topo_map.copy()
			enhanced_topo_map_img[nan_mask] = 0.0
			enhanced_topo_map_img = skimage.img_as_ubyte(1.0 - enhanced_topo_map_img)
			enhanced_topo_map_img[nan_mask] = 0

			skimage.io.imsave(save_path + 'enhanced_topo_maps/enhanced_topo_map_' + key + '_grey.png', enhanced_topo_map_img)

			enhanced_topo_map_img = enhanced_topo_map.copy()
			enhanced_topo_map_img[nan_mask] = 0.0
			enhanced_topo_map_img = skimage.img_as_ubyte(smap.to_rgba(enhanced_topo_map_img)[:,:,:3])
			enhanced_topo_map_img[nan_mask] = [0, 0, 0]

			skimage.io.imsave(save_path + 'enhanced_topo_maps/enhanced_topo_map_' + key + '.png', enhanced_topo_map_img)

			fig = plt.figure(figsize=(8,1))
			ax = plt.gca()
			cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=None, orientation='horizontal')
			cb.set_label('Topo Colorbar')
			plt.tight_layout()
			plt.savefig(save_path + 'enhanced_topo_maps/enhanced_topo_map_ ' + key + ' _colorbar.png')
			plt.close()

	print('Saving Texture and Blended Maps.')
	with warnings.catch_warnings():

		warnings.simplefilter('ignore')

		for key, enhanced_topo_map in enhanced_topo_maps.items():

			nm_min = -1.0
			nm_max = 1.0
			nan_mask = np.isnan(normal_map[:,:,0])

			texture_map_img = (normal_map - nm_min) / (nm_max - nm_min)
			texture_map_img = skimage.color.rgb2gray(texture_map_img)
		
			texture_map_img[nan_mask] = [0.0]
			texture_map_img = skimage.exposure.equalize_adapthist(
				texture_map_img,
				kernel_size=100,
				clip_limit=0.02
			)
			texture_map_img[nan_mask] = [0.0]
			texture_map_img = skimage.img_as_ubyte(texture_map_img)

			skimage.io.imsave(save_path + 'texture_map.png', texture_map_img)

			cmap = plt.cm.get_cmap(BLENDED_MAP_COLORMAP)
			smap = plt.cm.ScalarMappable(norm=None, cmap=cmap)
			nan_mask = np.isnan(enhanced_topo_map)
			texture_map_img[nan_mask] = 0

			blended_map_img = enhanced_topo_map.copy()
			blended_map_img[nan_mask] = 0.0
			blended_map_img = skimage.img_as_ubyte(1.0 - blended_map_img)
			blended_map_img[nan_mask] = 0

			alpha = 0.5

			blended_map_img = alpha * blended_map_img + (1 - alpha) * texture_map_img
			blended_map_img = blended_map_img.astype(np.uint8)
			skimage.io.imsave(save_path + 'blended_maps/blended_map_' + key + '_grey.png', blended_map_img)

			blended_map_img = enhanced_topo_map.copy()
			blended_map_img[nan_mask] = 0.0
			blended_map_img = skimage.img_as_ubyte(smap.to_rgba(blended_map_img)[:,:,:3])
			blended_map_img[nan_mask] = [0, 0, 0]

			alpha = 0.5

			blended_map_img = alpha * blended_map_img + (1 - alpha) * skimage.color.gray2rgb(texture_map_img)
			blended_map_img = blended_map_img.astype(np.uint8)
			skimage.io.imsave(save_path + 'blended_maps/blended_map_' + key + '.png', blended_map_img)




