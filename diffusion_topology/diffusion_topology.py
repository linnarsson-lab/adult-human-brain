# -*- coding: utf-8 -*-
import logging
import loompy
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import graph_tool.all as gt
from annoy import AnnoyIndex
from diffusion_topology import broken_stick

def knn_similarities(ds,  cells=None, n_genes=1000, k=50, annoy_trees=50, n_components=200, min_cells=10, min_nnz=2, mutual=True, metric='euclidean'):
	"""
	Compute knn similarity matrix for the given cells

	Args:
		ds (LoomConnecton):		Loom file
		cells (array of int):	Selection of cells that should be considered for the graph
		n_genes (int):			Number of genes to select
		k (int):				Number of nearest neighbours to search
		annoy_trees (int):		Number of Annoy trees used for kNN approximation
		n_components (int):		Number of principal components to use
		min_cells (int):		Minimum number of cells to retain as component (cluster)
		mutual (bool): 			If true, retain only mutual nearest neighbors
		metric (string):		Metric to use (euclidean or angular)

	Returns:
		knn (sparse matrix):	Matrix of similarities of k nearest neighbors
		genes (array of int):	Selection of genes that was used for the graph
		cells (array of int):	Selection of cells that are included in the graph
		sigma (numpy array):	Nearest neighbor similarity for each cell
	"""
	if cells == None:
		cells = np.array(range(ds.shape[1]))

	# Find the totals
	if not "_Total" in ds.col_attrs:
		ds.compute_stats()
	totals = ds.col_attrs["_Total"]
	median_cell = np.median(totals)

	# Compute an initial gene set
	logging.info("Selecting genes")
	ds.feature_selection(n_genes, method="SVR")
	genes = np.where(np.logical_and(ds.row_attrs["_Valid"] == 1, ds.row_attrs["_Excluded"] == 0))[0]

	# Pick a random set of (up to) 5000 cells
	logging.info("Subsampling cells")
	cells_sample = cells
	if len(cells_sample) > 5000:
		cells_sample = np.random.choice(cells_sample, size=5000, replace=False)
	cells_sample.sort()

	# Perform PCA based on the gene selection and the cell sample
	logging.info("Computing %d PCA components", n_components)
	pca = PCA(n_components=n_components)
	vals = ds[:, cells_sample][genes, :]
	vals = vals/totals[cells_sample]*median_cell
	vals = np.log(vals+1)
	vals = vals - np.mean(vals, axis=0)
	pca.fit(vals.transpose())

	bs = broken_stick(len(cells_sample), n_components)
	sig = pca.explained_variance_ratio_ > bs
	logging.info("Found %d significant principal components", np.sum(sig))

	logging.info("Creating approximate nearest neighbors model (annoy)")
	annoy = AnnoyIndex(n_components, metric=metric)
	for ix in cells:
		vals = ds[:, ix][genes, np.newaxis]
		vals = vals/totals[ix]*median_cell
		vals = np.log(vals+1)
		vals = vals - np.mean(vals)
		transformed = pca.transform(vals.transpose())[0]
		annoy.add_item(ix, transformed)

	annoy.build(annoy_trees)

	# Compute kNN and similarities for each cell, in sparse matrix IJV format
	logging.info("Computing mutual nearest neighbors")
	kth = int(max(k/10, 1))
	d = ds.shape[1]
	I = np.empty(d*k)
	J = np.empty(d*k)
	V = np.empty(d*k)
	sigma = np.empty((d,), dtype='float64') # The local kernel width
	for i in range(d):
		(nn, w) = annoy.get_nns_by_item(i, k, include_distances=True)
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

	if "_FakeDoublet" in ds.col_attrs:
		# Find and remove cells connected to doublets
		logging.info("Removing putative doublets")
		doublets = np.where(ds.col_attrs["_FakeDoublet"] == 1)[0]
		not_doublets = np.where(knn[doublets, :].sum(axis=1) == 0)[0]
		not_doublets = np.setdiff1d(not_doublets, doublets)
		ok_cells = np.intersect1d(ok_cells, not_doublets)

	logging.info("Done")
	return (knn, genes, ok_cells, sigma)

def block_model_graph(knn):
	"""
	From knn, make a graph-tool Graph object, a layout position list, and a stochastic blockmodel

	Args:
		knn (CSR sparse matrix):	knn adjacency matrix
	"""
	logging.info("Creating graph")
	g = gt.Graph(directed=False)
	knn = knn.tocoo()

	# Keep only half the edges, so the result is undirected
	sel = np.where(knn.row < knn.col)[0]
	g.add_vertex(n=knn.shape[0])
	g.add_edge_list(np.stack((knn.row[sel], knn.col[sel]), axis=1))
	logging.info("Creating graph layout")
	sfdp_pos = gt.sfdp_layout(g)
	logging.info("Fitting stochastic block model")
	block_state = gt.minimize_blockmodel_dl(g, deg_corr=True, overlap=True)
	return (g, sfdp_pos, block_state)

def _block_distances(d1, d2, n_components=20):
	# Calculate PCA keeping 20 components
	logging.info("Computing local PCA for (%d, %d) cells", d1.shape[0], d2.shape[0])
	pca = PCA(n_components=n_components)
	logging.info("d1 shape: (%d,%d)", d1.shape[0], d1.shape[1])
	logging.info("d2 shape: (%d,%d)", d2.shape[0], d2.shape[1])
	pca1 = pca.fit_transform(d1)
	logging.info("PCA1 shape: (%d,%d)", pca1.shape[0], pca1.shape[1])
	pca2 = pca.transform(d2)
	logging.info("PCA2 shape: (%d,%d)", pca2.shape[0], pca2.shape[1])

	# Calculate Euclidean distance matrix in PCA space
	logging.info("Computing local distance matrix in PCA space")
	dists = pairwise_distances(pca1, pca2, metric='euclidean')
	logging.info("dists shape: (%d,%d)", dists.shape[0], dists.shape[1])
	return dists

def knn_from_blocks(ds, cells, genes, state, k=50, min_cells=10):
	# Recompute the knn blockwise
	N = cells.shape[0]
	edges = state.get_matrix()
	logging.info("Number of edges: %d", state.get_matrix().nnz)
	block_index = np.array(state.get_majority_blocks().get_array().astype('int'))
	blocks = set(block_index)
	# Create empty sparse matrix to get started
	dists = sparse.coo_matrix((np.zeros(0), (np.zeros(0), np.zeros(0))), shape=(N, N))
	logging.info("Computing local distance matrices")
	for b1 in blocks:
		for b2 in blocks:
			if b1 == b2 or edges[b1, b2] > 0.0:
				logging.info("Loading data for (%d, %d)", b1, b2)
				d1 = np.log(ds[:, cells[block_index == b1]][genes, :]+1).transpose()
				d2 = np.log(ds[:, cells[block_index == b2]][genes, :]+1).transpose()
				m = sparse.coo_matrix(_block_distances(d1, d2))
				logging.info("Splicing submatrix into the main distance matrix")
				data = np.concatenate((dists.data, m.data))
				rows = np.concatenate((dists.row, m.row))
				cols = np.concatenate((dists.col, m.col))
				dists = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
				logging.info("Nonzero pairwise distances: %d", dists.nnz)


	# Shift the values to the negative range, so that zero is the greatest
	# distance (ensuring the sparse distance matrix is valid)
	logging.info("Computing local nearest neighbors")
	dists.data = dists.data.max() - dists.data
	dists = dists.tocsr()
	neigh = NearestNeighbors(n_jobs=-1, metric='precomputed')
	neigh.fit(dists)
	(d, nn) = neigh.kneighbors(n_neighbors=k, return_distance=True)

	# Convert to similarities
	logging.info("Converting to similarities")
	d = (d - d.min()) / -d.min()
	d = 1 - d

	# Turn this again into a sparse matrix, and make it mutual
	data = np.zeros(nn.shape[0]*nn.shape[1])
	rows = np.zeros(nn.shape[0]*nn.shape[1])
	cols = np.zeros(nn.shape[0]*nn.shape[1])
	for i in range(nn.shape[1]):
		rows[i*nn.shape[0]:(i+1)*nn.shape[0]] = dists.row
		cols[i*nn.shape[0]:(i+1)*nn.shape[0]] = nn[:, i]
		data[i*nn.shape[0]:(i+1)*nn.shape[0]] = d[:, i]

	knn = sparse.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
	# Make it mutual
	logging.info("Making it mutual")
	# This removes all edges that are not reciprocal
	knn = knn.minimum(knn.transpose())

	# Remove small disconnected components
	logging.info("Removing small disconnected components")
	(_, labels) = sparse.csgraph.connected_components(knn, directed='False')
	sizes = np.bincount(labels)
	cells = cells[(sizes > min_cells)[labels]]

	return (knn, genes, cells)



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

