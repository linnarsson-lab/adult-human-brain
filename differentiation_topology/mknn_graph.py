# -*- coding: utf-8 -*-
import logging
import loompy
import numpy as np
from typing import *
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors, BallTree
from sklearn.metrics import pairwise_distances
from scipy.special import polygamma
from sklearn.decomposition import PCA, IncrementalPCA
from annoy import AnnoyIndex
import networkx as nx
import differentiation_topology as dt


def knn_similarities(ds: loompy.LoomConnection, config: Dict, cells: np.ndarray, genes: np.ndarray) -> sparse.coo_matrix:
	"""
	Compute knn similarity matrix for the given cells

	Args:
		ds (LoomConnection):	Loom file
		config (dict):			A graph config dictionary
		cells (array of int):	Selection of cells that will be included in the graph
		genes (array of int):	Selection of genes that will be used for the graph

	Returns:
		knn (sparse matrix):	Matrix of similarities of k nearest neighbors
		sigma (numpy array):	Nearest neighbor similarity for each cell
	"""
	n_genes = config["n_genes"]
	k = config["k"]
	kth = config["kth"]
	use_radius = config["use_radius"]
	n_trees = config["n_trees"]
	search_k = config["search_k_factor"]*config["k"]*config["n_trees"]
	n_components = config["n_components"]
	stitch = config["stitch_components"]
	normalize = config["normalize"]
	standardize = config["standardize"]
	min_cells = config["min_cells"]
	mutual = config["mutual"]
	metric = config["metric"]
	filter_doublets = config["filter_doublets"]

	# Perform PCA based on the gene selection and the cell sample
	logging.info("Computing %d PCA components", n_components)
	pca = IncrementalPCA(n_components=n_components)

	cols_per_chunk = 5000
	logging.info("Fitting cells in batches of %d", cols_per_chunk)		
	ix = 0
	while ix < ds.shape[1]:
		cols_per_chunk = min(ds.shape[1] - ix, cols_per_chunk)
		selection = cells - ix
		# Pick out the cells that are in this batch
		selection = selection[np.where(np.logical_and(selection >= 0, selection < ix + cols_per_chunk))[0]]
		if selection.shape[0] == 0:
			continue
		# Load the whole chunk from the file, then extract genes and cells using fancy indexing
		vals = ds[:, ix:ix + cols_per_chunk][genes, :][:, selection]
		if normalize:
			vals = vals / totals[selection + ix] * median_cell
		vals = np.log(vals+1)
		vals = vals - np.mean(vals, axis=0)
		if standardize:
			vals = vals/np.std(vals, axis=0)
		pca.partial_fit(vals.transpose())

		ix = ix + cols_per_chunk

	bs = dt.broken_stick(len(genes), n_components)
	sig = pca.explained_variance_ratio_ > bs
	logging.info("Found %d significant principal components (but using all %d)", np.sum(sig), n_components)

	logging.info("Creating approximate nearest neighbors model (annoy)")
	annoy = AnnoyIndex(n_components, metric=metric)
	transformed = np.zeros((cells.shape[0],n_components))
	for ix, cell in enumerate(cells):
		vals = ds[:, cell][genes, np.newaxis]
		if normalize:
			vals = vals/totals[cell]*median_cell
		vals = np.log(vals+1)
		vals = vals - np.mean(vals)
		if standardize:
			vals = vals/np.std(vals, axis=0)
		transformed[ix,:] = pca.transform(vals.transpose())[0]
		annoy.add_item(cell, transformed[ix,:])

	# cols_per_chunk = 5000
	# ix = 0
	# while ix < ds.shape[1]:
	# 	cols_per_chunk = min(ds.shape[1] - ix, cols_per_chunk)
	# 	selection = cells - ix
	# 	# Pick out the cells that are in this batch
	# 	selection = selection[np.where(np.logical_and(selection >= 0, selection < ix + cols_per_chunk))[0]]
	# 	if selection.shape[0] == 0:
	# 		continue
	# 	# Load the whole chunk from the file, then extract genes and cells using fancy indexing
	# 	vals = ds[:, ix:ix + cols_per_chunk][genes, :][:, selection]
	# 	if normalize:
	# 		vals = vals/totals[selection + ix]*median_cell
	# 	vals = np.log(vals+1)
	# 	vals = vals - np.mean(vals, axis=0)
	# 	transformed = pca.transform(vals.transpose())
	# 	for col in range(transformed.shape[0]):
	# 		annoy.add_item(ix+col, transformed[col, :])

	# 	ix = ix + cols_per_chunk

	annoy.build(n_trees)

	# Compute kNN and similarities for each cell, in sparse matrix IJV format
	logging.info("Computing mutual nearest neighbors")
	d = ds.shape[1]
	I = np.empty(d*k)
	J = np.empty(d*k)
	V = np.empty(d*k)
	sigma = np.empty((d,), dtype='float64') # The local kernel width
	for i in range(d):
		(nn, w) = annoy.get_nns_by_item(i, k, include_distances=True, search_k=search_k)
		w = np.array(w)
		I[i*k:(i+1)*k] = [i]*k
		J[i*k:(i+1)*k] = nn
		V[i*k:(i+1)*k] = w
		sigma[i] = w[kth-1]

	if use_radius:
		# Compute a radius neighborhood based on the average local kernel width (sigma)
		radius = np.mean(sigma)
		logging.info("Mean distance to kth neighbor: " + str(radius))
		logging.info("Computing radius neighbors")
		ball_tree = BallTree(transformed)
		(J, V) = ball_tree.query_radius(transformed, r=radius, return_distance=True)
		I = []
		for ix,rn in enumerate(J):
			I.extend([ix]*rn.shape[0])
		J = np.concatenate(J).ravel()
		V = np.concatenate(V).ravel()
		I = np.array(I)


	# k nearest neighbours
	knn = sparse.coo_matrix((V, (I, J)), shape=(d, d))

	data = knn.data
	rows = knn.row
	cols = knn.col

	# Convert to similarities by rescaling and subtracting from 1
	data = data / data.max()
	data = 1 - data
	sigma = sigma / sigma.max()
	sigma = 1 - sigma

	knn = sparse.coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()

	if mutual:
		# Compute Mutual knn
		# This removes all edges that are not reciprocal
		knn = knn.minimum(knn.transpose())
	else:
		# Make all edges reciprocal
		# This duplicates all edges that are not reciprocal
		knn = knn.maximum(knn.transpose())

	# Find and remove disconnected components
	logging.info("Removing small components")
	(_, labels) = sparse.csgraph.connected_components(knn, directed='False')
	sizes = np.bincount(labels)
	ok_cells = np.where((sizes > min_cells)[labels])[0]
	logging.info("Small components removed (%d cells)", cells.shape[0] - ok_cells.shape[0])

	# if filter_doublets:
	# 	logging.warn("Doublet filtering is broken")
	# 	if not "_FakeDoublet" in ds.col_attrs:
	# 		logging.error("Cannot filter doublets, because no fake doublets were found")
	# 		return None
	# 	# Find and remove cells connected to doublets
	# 	logging.info("Removing putative doublets")
	# 	# List all the fake doublets
	# 	fake_doublets = np.where(ds.col_attrs["_FakeDoublet"] == 1)[0]
	# 	# Cells not connected to fake doublets
	# 	not_doublets = np.where(knn[fake_doublets].astype('bool').astype('int').sum(axis=0) <= 3)[1]
	# 	# Remove actual fake doublets
	# 	not_doublets = np.setdiff1d(not_doublets, fake_doublets)
	# 	logging.info("Removing %d putative doublets", ds.shape[1] - not_doublets.shape[0])
	# 	# Intersect with list of ok cells
	# 	ok_cells = np.intersect1d(ok_cells, not_doublets)
	# 	logging.info("Keeping %d cells in graph", ok_cells.shape[0])

	logging.info("Done")
	return (knn, genes, ok_cells, sigma)