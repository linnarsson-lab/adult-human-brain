import cytograph as cg
import numpy as np
import scipy.sparse as sparse
import loompy
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, NearestNeighbors
import logging
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from typing import *
import os
from umap import UMAP
from numba import jit


def stabilize(data: np.ndarray, size_factors: np.ndarray) -> np.ndarray:
	data = data * size_factors
	data = (np.sqrt(data) + 0.8 * np.sqrt(data + 1)) / 1.8
	return data


class Cytograph3:
	def __init__(self, accel: bool = False, log: bool = True, normalize: bool = True, n_genes: int = 500, n_factors: int = 100, a: float = 1, b: float = 10, c: float = 1, d: float = 10, min_k: int = 10, max_k: int = 100, max_d: float = 10, k_smoothing: int = 100, max_iter: int = 200) -> None:
		self.accel = accel
		self.log = log
		self.normalize = normalize
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_smoothing = k_smoothing
		self.min_k = min_k
		self.max_k = max_k
		self.max_d = max_d
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.max_iter = max_iter

	def fit(self, ds: loompy.LoomConnection) -> None:
		# Select genes
		logging.info("** STAGE 1 OF 3 **")
		logging.info("Calculating size factors")
		sums = ds.map([np.sum], axis=1)[0]
		size_factors = np.median(sums) / sums

		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		genes = np.sort(cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd))
		data = stabilize(ds[genes, :], size_factors).T

		# KNN
		logging.info(f"Calculating KNN")
		nn = NearestNeighbors(self.k_smoothing, algorithm="ball_tree", metric="minkowski", p=20, n_jobs=4)
		nn.fit(data)
		knn = nn.kneighbors_graph(data, mode='distance').tocoo()
		knn.setdiag(1)
		s = (knn.data < self.max_d) & (knn.data > 0)
		knn = sparse.coo_matrix(((knn.data[s] > 0).astype('int'), (knn.row[s], knn.col[s])))

		# Poisson smoothing (in place)
		logging.info(f"Poisson smoothing")
		ds["smoothened"] = 'int32'
		if "spliced" in ds.layers:
			ds["spliced_ps"] = 'int32'
			ds["unspliced_ps"] = 'int32'
		for (ix, indexes, view) in ds.scan(axis=0):
			if "spliced" in ds.layers:
				ds["spliced_ps"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced_ps"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["smoothened"][indexes.min(): indexes.max() + 1, :] = ds["spliced_ps"][indexes.min(): indexes.max() + 1, :] + ds["unspliced_ps"][indexes.min(): indexes.max() + 1, :]
			else:
				ds["smoothened"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T

		# Select genes
		logging.info("** STAGE 2 OF 3 **")
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False, layer="smoothened")
		normalizer.fit(ds)
		genes = np.sort(cg.FeatureSelection(self.n_genes, layer="smoothened").fit(ds, mu=normalizer.mu, sd=normalizer.sd))
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		data = stabilize(ds["smoothened"][genes, :], size_factors).T

		logging.info(f"Computing balanced KNN (k = {self.max_k}) in latent space")
		bnn = cg.BalancedKNN(k=self.max_k, metric="minkowski", minkowski_p=20, maxl=2 * self.max_k, sight_k=2 * self.max_k)
		bnn.fit(data)
		knn = bnn.kneighbors_graph(mode='distance').tocoo()
		s = (knn.data < self.max_d) & (knn.data > 0)
		knn = sparse.coo_matrix((knn.data[s], (knn.row[s], knn.col[s])))
		mknn = knn.minimum(knn.transpose())
		ds.col_graphs.KNN = knn
		ds.col_graphs.MKNN = mknn

		logging.info(f"UMAP embedding")
		umap = UMAP(metric="minkowski", metric_kwds={"p": 20}, min_dist=0.01, n_neighbors=25, n_components=2).fit_transform(data)
		ds.ca.UMAP = umap

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain()
		labels = pl.fit_predict(ds, "KNN", "UMAP")
		n_labels = np.max(labels) + 1
		ds.ca.Clusters = labels + 1
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {n_labels} clusters")

		logging.info("** STAGE 3 OF 3 **")
		logging.info("Marker selection")
		temp = ds.ca.Clusters
		ds.ca.Clusters = labels - labels.min()
		(genes, _, _) = cg.MarkerSelection(n_markers=int(self.n_genes / n_labels), findq=False).fit(ds)
		data = stabilize(ds["smoothened"][genes, :], size_factors).T

		logging.info(f"Computing balanced KNN (k = {self.max_k}) in latent space")
		bnn = cg.BalancedKNN(k=self.max_k, metric="minkowski", minkowski_p=20, maxl=2 * self.max_k, sight_k=2 * self.max_k)
		bnn.fit(data)
		knn = bnn.kneighbors_graph(mode='distance').tocoo()
		# truncate at max_d but leave at least min_k neighbors
		for i in range(knn.shape):
			items = np.where((knn.row == i) & (knn.data > 0))[0]
			ordering = np.argsort(knn.data[items])
			s = np.union1d(items[ordering][:self.min_k], items[knn.data[items] > self.max_d])
			knn.data[np.setdiff1d(items, s)] = 0
		nnz = (knn.data > 0)
		knn = sparse.coo_matrix((knn.data[nnz], (knn.row[nnz], knn.col[nnz])))
		mknn = knn.minimum(knn.transpose())
		ds.col_graphs.KNN = knn
		ds.col_graphs.MKNN = mknn

		logging.info(f"UMAP embedding")
		umap = UMAP(metric="minkowski", metric_kwds={"p": 20}, min_dist=0.1, n_neighbors=25, n_components=2).fit_transform(data)
		ds.ca.UMAP = umap

		logging.info(f"UMAP embedding in 3D")
		umap3 = UMAP(metric="minkowski", metric_kwds={"p": 20}, min_dist=0.1, n_neighbors=25, n_components=3).fit_transform(data)
		ds.ca.UMAP3 = umap3

		logging.info(f"tSNE embedding")
		tsne = TSNE(metric="euclidean", init=umap).fit_transform(data)
		ds.ca.TSNE = tsne

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain()
		labels = pl.fit_predict(ds, "KNN", "UMAP")
		n_labels = np.max(labels) + 1
		ds.ca.Clusters = labels + 1
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {n_labels} clusters")

		logging.info("** DONE **")


# TODO:
#
# Compute neighborhood enrichment
# Perform the variance-stabilizing transformation
# Set all but the top K enriched genes (for each cell) to zero
# Use Sanfransokyo distance in this space

@jit('double(double[:], double[:], int64)', nopython=True)
def sanfransokyodist(u, v, dof):
	f = (u.shape[0] - dof) / u.shape[0]
	l1_diff = np.abs(u - v)
	l1_sum = np.abs(u + v)
	zeros = l1_sum == 0
	dist = l1_diff / l1_sum
	dist[zeros] = 1
	return ((np.sum(dist) / u.shape[0]) - f) / (1 -f)