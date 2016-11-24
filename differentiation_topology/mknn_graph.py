# -*- coding: utf-8 -*-
import logging
import loompy
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from scipy.special import polygamma
from sklearn.decomposition import PCA, IncrementalPCA
import graph_tool.all as gt
from annoy import AnnoyIndex
import networkx as nx
import community
import differentiation_topology as dt


def knn_similarities(ds, config):
	"""
	Compute knn similarity matrix for the given cells

	Args:
		ds (LoomConnecton):		Loom file
		config (dict):			A graph config dictionary
	Returns:
		knn (sparse matrix):	Matrix of similarities of k nearest neighbors
		genes (array of int):	Selection of genes that was used for the graph
		cells (array of int):	Selection of cells that are included in the graph
		sigma (numpy array):	Nearest neighbor similarity for each cell
	"""
	cells = config["cells"]
	n_genes = config["n_genes"]
	k = config["k"]
	n_trees = config["n_trees"]
	search_k = config["search_k_factor"]*config["k"]*config["n_trees"]
	n_components = config["n_components"]
	normalize = config["normalize"]
	standardize = config["standardize"]
	min_cells = config["min_cells"]
	mutual = config["mutual"]
	metric = config["metric"]
	filter_doublets = config["filter_doublets"]

	if cells is None:
		cells = np.array(range(ds.shape[1]))
	cells = np.intersect1d(cells, np.where(ds.col_attrs["_Valid"] == 1)[0])

	# Find the totals
	if not "_Total" in ds.col_attrs:
		ds.compute_stats()
	totals = ds.col_attrs["_Total"]
	median_cell = np.median(totals)

	# Compute an initial gene set
	logging.info("Selecting genes")
	with np.errstate(divide='ignore',invalid='ignore'):
		ds.feature_selection(n_genes, method="SVR")
	if "_Valid" in ds.row_attrs:
		genes = np.where(np.logical_and(ds.row_attrs["_Valid"] == 1, ds.row_attrs["_Excluded"] == 0))[0]
	else:
		genes = np.where(ds.row_attrs["_Excluded"] == 0)[0]

	logging.info("Using %d genes", genes.shape[0])

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
			vals = vals/totals[selection + ix]*median_cell
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
	for ix in cells:
		vals = ds[:, ix][genes, np.newaxis]
		if normalize:
			vals = vals/totals[ix]*median_cell
		vals = np.log(vals+1)
		vals = vals - np.mean(vals)
		if standardize:
			vals = vals/np.std(vals, axis=0)
		transformed = pca.transform(vals.transpose())[0]
		annoy.add_item(ix, transformed)

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
	kth = int(max(k/10, 1))
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

def make_graph(knn, jaccard=False, cooling_step=0.95):
	"""
	From knn, make a graph-tool Graph object, a Louvain partitioning and a layout position list

	Args:
		knn (COO sparse matrix):	knn adjacency matrix
		jaccard (bool):				If true, replace knn edge weights with Jaccard similarities

	Returns:
		g (graph.tool Graph):		the Graph corresponding to the knn matrix
		labels (ndarray of int): 	Louvain partition label for each node in the graph
		sfdp (ndarray matrix):		X,Y position for each node
	"""
	logging.info("Creating graph")
	g = gt.Graph(directed=False)

	# Keep only half the edges, so the result is undirected
	sel = np.where(knn.row < knn.col)[0]
	logging.info("Graph has %d edges", sel.shape[0])

	g.add_vertex(n=knn.shape[0])
	edges = np.stack((knn.row[sel], knn.col[sel]), axis=1)
	g.add_edge_list(edges)
	w = g.new_edge_property("double")
	if jaccard:
		js = []
		knncsr = knn.tocsr()
		for i, j in edges:
			r = knncsr.getrow(i)
			c = knncsr.getrow(j)
			shared = r.minimum(c).nnz
			total = r.maximum(c).nnz
			js.append(shared/total)
		w.a = np.array(js)
	else:
		# use the input edge weights
		w.a = knn.data[sel]

	logging.info("Louvain partitioning")
	partitions = community.best_partition(nx.from_scipy_sparse_matrix(knn))
	labels = np.fromiter(partitions.values(), dtype='int')

	logging.info("Creating graph layout")
	#label_prop = g.new_vertex_property("int", vals=labels)
	sfdp = gt.sfdp_layout(g, eweight=w, epsilon=0.0001, cooling_step=cooling_step).get_2d_array([0, 1]).transpose()

	return (g, labels, sfdp)

# block_state = gt.minimize_blockmodel_dl(g, deg_corr=True, overlap=True)
# blocks = state.get_majority_blocks().get_array()


def sparse_dmap(m, sigma):
	"""
	Compute the diffusion map of a sparse similarity matrix

	Args:
		m (sparse.coo_matrix):	Sparse matrix of similarities in [0,1]
		sigma (numpy.array):    Array of nearest neighbor similarities

	Returns:
		tsym (sparse.coo_matrix):  Symmetric transition matrix T

	Note:
		The input is a matrix of *similarities*, not distances, which helps sparsity because very distant cells
		will have similarity near zero (whereas distances would be large).

	"""
	m = m.tocoo()

	# Convert sigma to distances
	sigma = 1 - sigma

	# The code below closely follows that of the diffusion pseudotime paper (supplementary methods)

	# sigma_x^2 + sigma_y^2 (only for the non-zero rows and columns)
	ss = np.power(sigma[m.row], 2) + np.power(sigma[m.col], 2)

	# In the line below, '1 - m.data' converts the input similarities to distances,
	# but only for the non-zero entries. That's fine because in the end (tsym) the
	# zero entries will end up zero anyway, so no need to involve them
	kxy = sparse.coo_matrix((np.sqrt(2*sigma[m.row]*sigma[m.col]/ss) * np.exp(-np.power(1-m.data, 2)/(2*ss)), (m.row, m.col)), shape=m.shape)
	zx = kxy.sum(axis=1).A1
	wxy = sparse.coo_matrix((kxy.data/(zx[kxy.row]*zx[kxy.col]), (kxy.row, kxy.col)), shape=kxy.shape)
	zhat = wxy.sum(axis=1).A1
	tsym = sparse.coo_matrix((wxy.data * np.power(zhat[wxy.row], -0.5) * np.power(zhat[wxy.col], -0.5), (wxy.row, wxy.col)), shape=wxy.shape)
	return tsym

def dpt(f, t, k, path_integral=True):
	"""
	Compute k steps of pseudotime evolution from starting vector f and transition matrix t

	Args:
		f (numpy 1D array):			The starting vector of N elements
		t (numpy sparse 2d array):	The right-stochastic transition matrix
		k (int):					Number of steps to evolve the input vector
		path_integral (bool):		If true, calculate all-paths integral; otherwise calculate time evolution

	Note:
		This algorithm is basically the same as Google PageRank, except we start typically from
		a single cell (or a few cells), rather than a uniform distribution, and we don't make
		random jumps.
		See: http://michaelnielsen.org/blog/using-your-laptop-to-compute-pagerank-for-millions-of-webpages/
		and the original PageRank paper: http://infolab.stanford.edu/~backrub/google.html
	"""
	f = sparse.csr_matrix(f)
	result = np.zeros(f.shape)
	if path_integral:
		for _ in range(k):
			f = f.dot(t)
			result = result + f
		return result.A1
	else:
		for _ in range(k):
			f = f.dot(t)
		return f.A1

#
# Example of how to use diffusion pseudotime
#
# root = np.zeros(ds.shape[1]) # +(1.0/ds.shape[1])
# root[0] = 1
# t = dt.sparse_dmap(knn, sigma)

# # Iterate cumulative pseudotime from the root cells
# f = dt.dpt(root, t, 1000)
#
# Now f contains the probability distribution after 1000 diffusion time steps