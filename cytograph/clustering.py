from typing import *
import os
from shutil import copyfile
import numpy as np
import random
import logging
import igraph
import cytograph as cg
import loompy
import logging
from scipy import sparse
from scipy.special import polygamma
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from scipy.stats import ks_2samp
import networkx as nx
import hdbscan
from sklearn.cluster import DBSCAN


class Clustering:
	def __init__(self, method: str, outliers: bool = True, eps_pct: float = 65, min_pts: int = None) -> None:
		"""
		Args:
			method		'hdbscan', 'dbscan', or 'lj'
			outliers	True to allow outliers
		"""
		self.method = method
		self.outliers = outliers
		self.eps_pct = eps_pct
		self.min_pts = min_pts

	def fit_predict(self, ds: loompy.LoomConnection) -> np.ndarray:
		n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
		n_total = ds.shape[1]
		logging.info("%d of %d cells were valid", n_valid, n_total)
		logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])
		cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

		if self.method == "hdbscan":
			logging.info("HDBSCAN clustering in t-SNE space")
			min_pts = 10 if n_valid < 3000 else (20 if n_valid < 20000 else 100)
			tsne_pos = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[cells, :]
			clusterer = hdbscan.HDBSCAN(min_cluster_size=min_pts)
			labels = clusterer.fit_predict(tsne_pos)
		elif self.method == "dbscan":
			logging.info("DBSCAN clustering in t-SNE space")
			if self.min_pts is None:
				self.min_pts = 10 if n_valid < 3000 else (20 if n_valid < 20000 else 100)
			tsne_pos = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[cells, :]

			# Determine a good epsilon
			nn = NearestNeighbors(n_neighbors=self.min_pts, algorithm="ball_tree", n_jobs=4)
			nn.fit(tsne_pos)
			knn = nn.kneighbors_graph(mode='distance')
			k_radius = knn.max(axis=1).toarray()
			epsilon = np.percentile(k_radius, self.eps_pct)

			clusterer = DBSCAN(eps=epsilon, min_samples=self.min_pts)
			labels = clusterer.fit_predict(tsne_pos)
			if not self.outliers:
				# Assign each outlier to the same cluster as the nearest non-outlier
				nn = NearestNeighbors(n_neighbors=50, algorithm="ball_tree")
				nn.fit(tsne_pos[labels >= 0])
				nearest = nn.kneighbors(tsne_pos[labels == -1], n_neighbors=1, return_distance=False)
				labels[labels == -1] = labels[labels >= 0][nearest.flat[:]]
		elif self.method == "multilev":
			logging.info("comunity-multilevel clustering on unweighted KNN graph")
			(a, b, w) = ds.get_edges("KNN", axis=1)
			# knn = sparse.coo_matrix((w, (a, b)), shape=(ds.shape[1], ds.shape[1])).tocsr()[cells, :][:, cells]
			# sources, targets = knn.nonzero()
			G = igraph.Graph(n_total, list(zip(a, b)), directed=False)
			VxCl = G.community_multilevel(return_levels=False)
			labels = np.array(VxCl.membership)
		elif self.method == "wmultilev":
			logging.info("comunity-multilevel clustering on the multiscale KNN graph")
			(a, b, w) = ds.get_edges("KNN", axis=1)
			# knn = sparse.coo_matrix((w, (a, b)), shape=(ds.shape[1], ds.shape[1])).tocsr()[cells, :][:, cells]
			# a, b = knn.nonzero()
			G = igraph.Graph(n_total, list(zip(a, b)), directed=False, edge_attrs={'weight': w})
			VxCl = G.community_multilevel(return_levels=False, weights="weight")
			labels = np.array(VxCl.membership)
		elif self.method == "mknn_louvain":
			logging.info("comunity-multilevel clustering on the multiscale MKNN graph")
			(a, b, w) = ds.get_edges("MKNN", axis=1)
			random.seed(13)
			igraph._igraph.set_random_number_generator(random)
			G = igraph.Graph(n_total, list(zip(a, b)), directed=False, edge_attrs={'weight': w})
			VxCl = G.community_multilevel(return_levels=False, weights="weight")
			labels = np.array(VxCl.membership)
			logging.info(f"labels.shape = {labels.shape}")
			if not self.outliers:
				bigs = np.where(np.bincount(labels) >= 0)[0]
			else:
				bigs = np.where(np.bincount(labels) >= self.min_pts)[0]	
			mapping = {k: v for v, k in enumerate(bigs)}
			labels = np.array([mapping[x] if x in bigs else -1 for x in labels])						
		else:
			logging.info("Louvain clustering on the multiscale KNN graph")
			(a, b, w) = ds.get_edges("KNN", axis=1)
			knn = sparse.coo_matrix((w, (a, b)), shape=(ds.shape[1], ds.shape[1])).tocsr()[cells, :][:, cells]
			lj = cg.LouvainJaccard(resolution=1, jaccard=False)
			labels = lj.fit_predict(knn.tocoo())

		# At this point, cells should be labeled 0, 1, 2, ...
		# But there may also be cells labelled -1 for outliers, which we want to keep track of
		labels_all = np.zeros(ds.shape[1], dtype='int')
		outliers = np.zeros(ds.shape[1], dtype='int')
		labels_all[cells] = labels
		outliers[labels_all == -1] = 1
		labels_all[cells] = labels - np.min(labels)
		ds.ca.Clusters = labels_all
		ds.ca.Outliers = outliers
		logging.info("Found " + str(max(labels_all) + 1) + " clusters")
		if not len(set(ds.ca.Clusters)) == ds.ca.Clusters.max() + 1:
			raise ValueError("There are holes in the cluster ID sequence!")
		return labels_all
