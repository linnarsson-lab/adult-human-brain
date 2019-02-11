"""This module runs a simple velocity simulation in an euclidean space that is:
it will sample point in the direction of RNA velocity.
The modules calculates statistics on the collsions between the simulated moving points
and the original clouds of points.

# Usage example

## Run the simulation
knn_list, new_pcs_t_s = collision_simulator(pcs, pcs_t)

## To get stats with different level of granularity
full_stats = collision_stats(knn_list, vlm2.cluster_ix)
in_time, out_time = summ_stats(full_stats)
collision_matrix = transition_stats(in_time, cluster_ixs)

## To visualize the simulation
cluster_assigned = prep_collisions_vis(full_stats, cluster_ixs)
collisions_plot(pcs, pcs_t, new_pcs_t_s, cluster_assigned, cluster_ixs=cluster_ixs)
"""

import numpy as np
from scipy.stats import norm
from scipy.sparse import csr_matrix
from numba import njit, jit
from typing import List, Tuple
import ipyvolume as ipv
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def collision_simulator(pcs: np.ndarray, pcs_t: np.ndarray, steps: int=30, sight: float=1.5,
						k: int=50, scale: float=0.05, t_steps: np.ndarray=None, skip_zero_t: bool=False) -> Tuple[List[csr_matrix], np.ndarray]:
	"""Extrapulate in a linear, magnitude independent way
	the positions of cells in pca in successive steps of time
	
	Arguments
	---------
	
	pcs:  np.ndarray
		The pca of the original points
	
	pcs_t:  np.ndarray
		The pca of the points estimate after a time t so that the delta is the velocity
	
	steps: int, default = 30
		the number of grids steps to calculate into the future
		
	sight: float, default=1.5
		How far to project into the future
		Where the distance of the 2 farter points of the dataset is ~1
		
	k: int
		number of neighbours to look at to determine collsion
		
	scale: float, default=0.05
		Scale of the gaussian kernel to use
		as relative to the diagonal of the a bounding-box of the data
		
	t_steps: np.ndarray
		In alternative of steps and sight one can pass an array
		
	skip_zero_t: bool
		Wheterh to skip the starting position in the simulation
		
	Returns
	-------
	
	knn_list: List[np.ndarray]
		For each of the extrapolations points (i.e. steps)
		A sparse matrix that for every cells contains the value of the kernel contribution of each of its neighbours
	
	new_pcs_t_s: np.ndarray (3d)
		The postion of the extrapolation at each successive point in time
	
	"""
	
	# deltas = pcs_t - pcs
	if t_steps is None:
		t_steps = np.linspace(0, sight, steps)
		if skip_zero_t:
			t_steps = t_steps[1:]
			
	# Rescaling the axis to (1,1,1) makes the choices of steplenght easier
	occupied_cube = pcs.max(0) - pcs.min(0)
	rescaling_factors = 1 / occupied_cube
	pcsR, pcs_tR = rescaling_factors * pcs, rescaling_factors * pcs_t
	
	delta = pcs_tR - pcsR
	delta = delta / np.linalg.norm(delta, 2, 1)[:, None]
	
	new_pcs_t_list = []
	for i in range(len(t_steps)):
		new_pcs_t = (pcsR + delta * t_steps[i]) * occupied_cube
		new_pcs_t_list.append(new_pcs_t)
		
	# Calculate the characteristic distance = diagonal of the cube
	char_dist = np.linalg.norm(pcs.max(0) - pcs.min(0))
	
	nn = NearestNeighbors(k, n_jobs=30)
	nn.fit(pcs)
	knn_list = []
	for new_pcs_t in new_pcs_t_list:
		knn = nn.kneighbors_graph(new_pcs_t, mode="distance")
		# Note the kernel is isotropic (this might be inappropriate since different pcs have different scales)
		knn.data = np.exp(norm.logpdf(knn.data, 0, char_dist * scale) - norm.logpdf(0, 0, char_dist * scale))
		knn_list.append(knn)
		
	return knn_list, np.stack(new_pcs_t_list, -1)


def collision_stats(knn_list: List[csr_matrix], cluster_ixs: np.ndarray) -> np.ndarray:
	"""Reduce the collision silmulation to densities tracks
	
	Arguments
	---------
	
	knn_list: List[np.ndarray]
		This is the first output of collision_simulator
		For each of the extrapolations points (i.e. steps)
		A sparse matrix that for every cells contains the value of the kernel contribution of each of its neighbours
	
	cluster_ixs: np.ndarray
		The cluster indexes of each cells, the entries should go from 0 to N-1 when N are the clusters
		
	Returns
	-------
	
	fulls stats: np.ndarray shape=(clusters, timepoints, cells)
		An array containing the values for the normalized kernel density by cell, by timepoint by cluster
		
	"""
	k = (knn_list[0][0, :] > 0).sum()
	clusters_uid = np.unique(cluster_ixs)
	indicators = [cluster_ixs == uid for uid in clusters_uid]
	full_stats = np.ndarray((len(clusters_uid), len(knn_list), len(cluster_ixs)))
	for i in range(len(clusters_uid)):
		for j in range(len(knn_list)):
			full_stats[i, j, :] = knn_list[j].dot(indicators[i]) / k
			
	return full_stats


@njit
def summ_stats(full_stats: np.ndarray, thresh: float=0.5) -> Tuple[np.ndarray, np.ndarray]:
	"""Summarize the full temporal stats with two matrix indicating the
	time at which each cells enters in each cluster
	
	Arguments
	---------
	full_stats: np.ndarray  shape=(clusters, timepoints, cells)
		The output of the function collision_stats
		An array containing the values for the normalized kernel density by cell, by timepoint by cluster
	thresh: float (default=0.5)
		The threshold to apply to the density matrix
		
	Returns
	-------
	in_time: np.ndarray (clusters, cells)
		The time step at what each cells enter in each cluster area. -1 if it never does
	
	out_time
		The time step at what each cells exits in each cluster area. -1 if it never enters or enters and never exits
	
	"""
	in_time = np.full((full_stats.shape[0], full_stats.shape[2]), -1)
	out_time = np.full((full_stats.shape[0], full_stats.shape[2]), -1)
	# For each cell
	for k in range(full_stats.shape[2]):
		# For cluster
		for i in range(full_stats.shape[0]):
			temp = full_stats.shape[1]
			# For each step in time until found, search for cluster contact
			for j in range(full_stats.shape[1]):
				if full_stats[i, j, k] > thresh:
					in_time[i, k] = j
					temp = j
					break
			# After contact look for cluster end-of contact
			for j in range(temp, full_stats.shape[1]):
				if full_stats[i, j, k] < thresh:
					out_time[i, k] = j
					break
	return in_time, out_time


def transition_stats(in_time: np.ndarray, cluster_ixs: np.ndarray) -> np.ndarray:
	"""Statistics on the transition matrix
	
	Arguments
	---------
	in_time: np.ndarray (clusters, cells)
		The outputs of summ_stats
		The time step at what each cells enter in each cluster area. -1 if it never does
		
	Returns
	-------
	transition_matrix: np.ndarray (clusters, clusters)
		Contain the fraction of cells of a certain cluster that during the trajectiory gets to collide with another cluster
		The diagonal contains the fraction of clusters that are ever positive for collision with their original cluster
		rows: cluster-trom , columns: cluster-to
	
	Note
	----
	The diagonal value will not be one depending on the overalp of the two populations
	"""
	return np.stack([(in_time[:, cluster_ixs == i] >= 0).mean(1) for i in np.unique(cluster_ixs)])
			
			
def prep_collisions_vis(full_stats: np.ndarray, cluster_ixs: np.ndarray, thresh: float=0.5) -> np.ndarray:
	""" Returns for the simulation a label of the assigment
	
	Arguments
	---------
	full_stats: np.ndarray  shape=(clusters, timepoints, cells)
		The output of the function collision_stats
		An array containing the values for the normalized kernel density by cell, by timepoint by cluster
		
	cluster_ixs: np.ndarray
		The cluster indexes of each cells, the entries should go from 0 to N-1 when N are the clusters
		
	thresh: float (default=0.5)
		The threshold to apply to the density matrix
	
	Returns
	-------
	cluster_assigned: np.ndarray shape=(timepoints, cells)
		A matrix containing the cluster assigned at each simulation point at for each cell
		
	"""
	cluster_assigned = np.full((full_stats.shape[1], full_stats.shape[2]), -1)
	for i in np.unique(cluster_ixs):
		cluster_assigned[full_stats[i, :, :] > thresh] = i
	non_unique = (full_stats > thresh).sum(0) > 1
	cluster_assigned[non_unique] = i + 1
	return cluster_assigned
	

def collisions_plot(pcs: np.ndarray, pcs_t: np.ndarray, new_pcs_t_s: np.ndarray,
					cluster_assigned: np.ndarray, colorandum: np.ndarray=None, cluster_ixs: np.ndarray=None) -> None:
	"""Utility function to plot in 3d an animation of the simulation.
	
	Arguments
	---------
	pcs: np.ndarray
		The position of the cells in the 3d pca
		
	pcs_t: np.ndarray
		The position of the cells in the 3d pca extrapolated at time t
		
	new_pcs_t_s: np.ndarray
		The second output of collision_simulator
		The position as resulting from the simulation
		
	cluster_assigned: np.ndarray
		The output of prep_collisions_vis
		The cluster assigned to each of the simulation points
		
	colorandum: np.ndarray (default=None) shape=(cells, 3)
		An array containing the color of each of the cells
		If None the colors from plt.cm.Set1 will be used but cluster_ixs should be provided

	cluster_ixs: np.ndarray,
		The cluster labels for each cell
		if None it will try to determine everything from colorandum and cluster_assigned
	
	Returns
	-------
	
	Nothing but it plots a ipyvolume animated graph
	
	
	"""
	ipv.figure()
	if colorandum is None:
		if cluster_ixs is None:
			raise ValueError("Either colorandum or cluster_ixs should be provided.")
		colorandum = plt.cm.Set1_r(cluster_ixs)
		
	vv2 = ipv.quiver(pcs[:, 0], pcs[:, 1], pcs[:, 2],
					 pcs_t[:, 0] - pcs[:, 0], pcs_t[:, 1] - pcs[:, 1], pcs_t[:, 2] - pcs[:, 2],
					 size=4, color=colorandum)  # marker='sphere'
	
	if cluster_ixs is None:
		colors_allowed = plt.cm.tab10(np.arange(len(np.unique(cluster_assigned))))[:, :3]
	else:
		colors_allowed = plt.cm.tab10(np.arange(len(np.unique(cluster_ixs)) + 2))[:, :3]
	colors_allowed[-1, :] = 0, 0, 0
	colors = colors_allowed[cluster_assigned]
	
	ss2 = ipv.scatter(new_pcs_t_s[:, 0, :].T, new_pcs_t_s[:, 1, :].T, new_pcs_t_s[:, 2, :].T,
					  size=2, color=colors, marker='sphere')
	ipv.xlim(pcs[:, 0].min(), pcs[:, 0].max())
	ipv.ylim(pcs[:, 1].min(), pcs[:, 1].max())
	ipv.zlim(pcs[:, 2].min(), pcs[:, 2].max())
	ipv.animation_control(ss2, interval=200)
	ipv.show()