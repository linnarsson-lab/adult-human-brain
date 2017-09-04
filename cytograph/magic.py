from typing import *
import os
import numpy as np
import logging
import cytograph as cg
import loompy
from scipy import sparse
from scipy.special import polygamma
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from scipy.stats import ks_2samp
import networkx as nx
import luigi


# Config classes should be camel cased
class magic(luigi.Config):
	n_genes = luigi.IntParameter(default=2000)
	standardize = luigi.BoolParameter(default=False)
	n_components = luigi.IntParameter(default=50)
	k = luigi.IntParameter(default=20)
	kth = luigi.IntParameter(default=2)
	t = luigi.IntParameter(default=2)


def magic_imputation(ds: loompy.LoomConnection, out_file: str) -> None:
	n_cells = ds.shape[1]

	logging.info("Normalization")
	normalizer = cg.Normalizer(magic().standardize)
	normalizer.fit(ds)

	logging.info("Selecting %d genes", magic().n_genes)
	genes = cg.FeatureSelection(magic().n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)

	logging.info("PCA projection")
	pca = cg.PCAProjection(genes, max_n_components=magic().n_components)
	transformed = pca.fit_transform(ds, normalizer)

	logging.info("Generating KNN graph")
	nn = NearestNeighbors(n_neighbors=magic().k, algorithm="ball_tree", n_jobs=4)
	nn.fit(transformed)
	knn = nn.kneighbors_graph(transformed, mode='distance')
	knn = knn.tocoo()

	# Convert to similarities by rescaling and subtracting from 1
	data = knn.data
	data = data / data.max()
	data = 1 - data
	knn = sparse.coo_matrix((data, (knn.row, knn.col)), shape=(n_cells, n_cells))

	logging.info("Determining adaptive kernel size for MAGIC")
	(dists, _) = nn.kneighbors(n_neighbors=magic().kth)
	sigma = dists[:, -1]
	sigma = sigma / sigma.max()

	tsym = sparse_dmap(knn, sigma).tocsr()

	# Impute the expression values for all genes, via the Markov diffusion process
	logging.info("Loading data for MAGIC imputation")
	original = ds[:, :]
	imputed = np.empty_like(original)
	logging.info("Computing MAGIC imputation")
	tsym_pow = tsym ** magic().t
	for ix in range(imputed.shape[0]):
		if original[ix, :].max() > 0:
			temp = tsym_pow.dot(original[ix, :])
			imputed[ix, :] = temp / temp.max() * np.percentile(original[ix, :], 99)
		else:
			imputed[ix, :] = original[ix, :]

	logging.info("Saving")
	loompy.create(out_file, imputed, ds.row_attrs, ds.col_attrs)


def sparse_dmap(m: sparse.coo_matrix, sigma: np.ndarray) -> sparse.coo_matrix:
	"""
	Compute the diffusion map of a sparse similarity matrix
	Args:
		m (sparse.coo_matrix):	Sparse matrix of similarities in [0,1]
		sigma (numpy.array):    Array of local kernel widths (distances to kth neighbor) in [0,1]
	Returns:
		tsym (sparse.coo_matrix):  Symmetric transition matrix T
	Note:
		The input is a matrix of *similarities*, not distances, which helps sparsity because very distant cells
		will have similarity near zero (whereas distances would be large).
	"""

	# The code below closely follows that of the diffusion pseudotime paper (supplementary methods)

	# sigma_x^2 + sigma_y^2 (only for the non-zero rows and columns)
	ss = np.power(sigma[m.row], 2) + np.power(sigma[m.col], 2)

	# In the line below, '1 - m.data' converts the input similarities to distances, but only for the non-zero entries
	# That's fine because in the end (tsym) the zero entries will end up zero anyway, so no need to involve them
	kxy = sparse.coo_matrix((np.sqrt(2 * sigma[m.row] * sigma[m.col] / ss) * np.exp(-np.power(1 - m.data, 2) / (2 * ss)), (m.row, m.col)), shape=m.shape)
	zx = kxy.sum(axis=1).A1
	wxy = sparse.coo_matrix((kxy.data / (zx[kxy.row] * zx[kxy.col]), (kxy.row, kxy.col)), shape=kxy.shape)
	zhat = wxy.sum(axis=1).A1
	tsym = sparse.coo_matrix((wxy.data * np.power(zhat[wxy.row], -0.5) * np.power(zhat[wxy.col], -0.5), (wxy.row, wxy.col)), shape=wxy.shape)
	return tsym


def bhd(f: np.ndarray, m: np.ndarray, t: int) -> np.ndarray:
	"""
	Compute k steps of backwards heat diffusion from starting vector f and transition matrix t
	Args:
		f (numpy 1D array):			The starting vector of N elements
		m (numpy sparse 2d array):	The right-stochastic transition matrix
		t (int):					Number of steps of back-diffusion
		path_integral (bool):		If true, calculate all-paths integral; otherwise calculate time evolution 
		"""
	f = sparse.csr_matrix(f)
	return np.linalg.matrix_power(m, t).dot(f)


def dpt(f: np.ndarray, t: np.ndarray, k: int, path_integral: bool = True) -> np.ndarray:
	"""
	Compute k steps of pseudotime evolution from starting vector f and transition matrix t
	Args:
		f (numpy 1D array):			The starting vector of N elements
		t (numpy sparse 2d array):	The right-stochastic transition matrix
		k (int):					Number of steps to evolve the input vector
		path_integral (bool):		If true, calculate all-paths integral; otherwise calculate time evolution 
	
	Note:
		This algorithm is basically the same as Google PageRank, except we start typically from 
		a single cell (or a few cells), rather than a uniform distribution, and we don't make random jumps.
		See: http://michaelnielsen.org/blog/using-your-laptop-to-compute-pagerank-for-millions-of-webpages/ and 
		the original PageRank paper: http://infolab.stanford.edu/~backrub/google.html
	"""
	f = sparse.csr_matrix(f)
	result = np.zeros(f.shape)
	if path_integral:
		for ix in range(k):
			f = f.dot(t)
			result = result + f
		return result.A1
	else:
		for ix in range(k):
			f = f.dot(t)
		return f.A1
