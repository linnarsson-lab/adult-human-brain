import logging

import community
import leidenalg
import networkx as nx
import igraph as ig
import numpy as np
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

import loompy


def is_outlier(points: np.ndarray, thresh: float = 3.5) -> np.ndarray:
	"""
	Returns a boolean array with True if points are outliers and False
	otherwise.

	Parameters:
	-----------
		points : An numobservations by numdimensions array of observations
		thresh : The modified z-score to use as a threshold. Observations with
			a modified z-score (based on the median absolute deviation) greater
			than this value will be classified as outliers.

	Returns:
	--------
		mask : A numobservations-length boolean array.

	References:
	----------
		Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
		Handle Outliers", The ASQC Basic References in Quality Control:
		Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
	"""
	if len(points.shape) == 1:
		points = points[:, None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation

	return modified_z_score > thresh


def kneighbors_graph(indices: np.ndarray, distances: np.ndarray, mode: str = 'distance'):

	# Adapted from scikit-learn kneighbors_graph (2022-02-15)
	# https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09b/sklearn/neighbors/_graph.py#L38

	n_queries = indices.shape[0]
	n_neighbors = indices.shape[1]

	if mode == "connectivity":
		data = np.ones(n_queries * n_neighbors)

	elif mode == "distance":
		data = np.ravel(distances)

	n_entries = n_queries * n_neighbors
	indptr = np.arange(0, n_entries + 1, n_neighbors)

	kneighbors_graph = csr_matrix(
		(data, indices.ravel(), indptr), shape=(n_queries, n_queries)
	)

	return kneighbors_graph


class PolishedLouvain:
	def __init__(self, resolution: float = 1.0, outliers: bool = True, min_cells: int = 10, graph: str = "MKNN", embedding: str = "TSNE", method: str = "python-louvain") -> None:
		self.resolution = resolution
		self.outliers = outliers
		self.min_cells = min_cells
		self.graph = graph
		self.embedding = embedding
		self.method = method  # "leiden" or "python-louvain"

	def _break_cluster(self, embedding: np.ndarray) -> np.ndarray:
		"""
		If needed, split the cluster by density clustering on the embedding

		Returns:
			An array of cluster labels (all zeros if cluster wasn't split)
			Note: the returned array may contain -1 for outliers
		"""
		# Find outliers in either dimension using Grubbs test
		xy = PCA().fit_transform(embedding)
		x = xy[:, 0]
		y = xy[:, 1]
		# Standardize x and y (not sure if this is really necessary)
		x = (x - x.mean()) / x.std()
		y = (y - y.mean()) / y.std()
		xy = np.vstack([x, y]).transpose()

		outliers = np.zeros(embedding.shape[0], dtype='bool')
		for _ in range(5):
			outliers[~outliers] = is_outlier(x[~outliers])
			outliers[~outliers] = is_outlier(y[~outliers])

		# See if the cluster is very dispersed
		min_pts = min(50, min(x.shape[0] - 1, max(5, round(0.1 * x.shape[0]))))
		if xy.shape[0] < 1000:
			nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
			nn.fit(xy)
			knn = nn.kneighbors_graph(mode='distance')
		else:
			nn = NNDescent(data=xy, n_jobs=-1, random_state=0)
			indices, distances = nn.query(xy, k=min_pts + 1)
			knn = kneighbors_graph(indices, distances, mode='distance')
		k_radius = knn.max(axis=1).toarray()
		epsilon = np.percentile(k_radius, 70)
		# Not too many outliers, and not too dispersed
		if outliers.sum() <= 3 and (np.sqrt(x**2 + y**2) < epsilon).sum() >= min_pts * 0.5:
			return np.zeros(embedding.shape[0], dtype='int')

		# Too many outliers, or too dispersed
		clusterer = DBSCAN(eps=epsilon, min_samples=round(min_pts * 0.5))
		return clusterer.fit_predict(xy)

	def fit_predict(self, ds: loompy.LoomConnection) -> np.ndarray:
		"""
		Given a sparse adjacency matrix, perform Louvain clustering, then polish the result

		Args:
			ds		The loom dataset
			graph	The graph to use

		Returns:
			labels:	The cluster labels (where -1 indicates outliers)

		"""
		if self.embedding in ds.ca:
			xy = ds.ca[self.embedding]
		else:
			raise ValueError(f"Embedding '{self.embedding}' not found in file")
		
		logging.info(f"Using {self.graph} graph")
		knn = ds.col_graphs[self.graph]

		logging.info("Louvain community detection")
		if self.method == "leiden":
			_, components = connected_components(knn)
			next_label = 0
			all_labels = np.zeros(ds.shape[1], dtype="int")
			for cc in np.unique(components):
				# Create an RNN graph restricted to the component
				cells = np.where(components == cc)[0]
				n_cells = cells.shape[0]
				if n_cells > self.min_cells:
					cc_rnn = ds.col_graphs[cells].RNN.tocsr()
					sources, targets = cc_rnn.nonzero()
					weights = cc_rnn[sources, targets]
					g = ig.Graph(list(zip(sources, targets)), directed=False, edge_attrs={'weight': weights})
					labels = np.array(leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, n_iterations=-1).membership) + next_label
				else:
					labels = np.zeros(n_cells) + next_label
				next_label = labels.max() + 1
				all_labels[cells] = labels
			labels = all_labels
		else:
			g = nx.from_scipy_sparse_matrix(knn)
			partitions = community.best_partition(g, resolution=self.resolution, randomize=False, random_state=0)
			labels = np.array([partitions[key] for key in range(knn.shape[0])])

		# Mark outliers using DBSCAN
		logging.info("Using DBSCAN to mark outliers")
		nn = NNDescent(data=xy, n_jobs=-1, random_state=0)
		indices, distances = nn.query(xy, k=11)
		knn = kneighbors_graph(indices, distances, mode='distance')
		k_radius = knn.max(axis=1).toarray()
		epsilon = np.percentile(k_radius, 80)
		clusterer = DBSCAN(eps=epsilon, min_samples=10)
		outliers = (clusterer.fit_predict(xy) == -1)
		labels[outliers] = -1

		# Mark outliers as cells in bad neighborhoods
		logging.info("Using neighborhood to mark outliers")
		nn = NNDescent(data=xy, n_jobs=-1, random_state=0)
		indices, distances = nn.query(xy, k=11)
		neighborhood = labels[indices] == labels[:, None]
		labels[neighborhood.sum(axis=1) / neighborhood.shape[1] < 0.2] = -1

		# Renumber the clusters
		retain = sorted(list(set(labels)))
		d = dict(zip(retain, np.arange(-1, len(set(retain)))))
		labels = np.array([d[x] if x in d else -1 for x in labels])

		# Break clusters based on the embedding
		logging.info("Breaking clusters")
		max_label = 0
		labels2 = np.copy(labels)
		for lbl in range(labels.max() + 1):
			cluster = labels == lbl
			if cluster.sum() < 10:
				continue
			adjusted = self._break_cluster(xy[cluster, :])
			new_labels = np.copy(adjusted)
			for i in range(np.max(adjusted) + 1):
				new_labels[adjusted == i] = i + max_label
			max_label = max_label + np.max(adjusted) + 1
			labels2[cluster] = new_labels
		labels = labels2

		# Set the local cluster label to the local majority vote
		logging.info("Smoothing cluster identity on the embedding")
		nn = NNDescent(data=xy, n_jobs=-1, random_state=0)
		indices, distances = nn.query(xy, k=11)
		labels = mode(labels[indices], axis=1)[0].flatten()

		# Mark tiny clusters as outliers
		logging.info("Marking tiny clusters as outliers")
		ix, counts = np.unique(labels, return_counts=True)
		labels[np.isin(labels, ix[counts < self.min_cells])] = -1

		# Renumber the clusters (since some clusters might have been lost in poor neighborhoods)
		retain = list(set(labels))
		if -1 not in retain:
			retain.append(-1)
		retain = sorted(retain)
		d = dict(zip(retain, np.arange(-1, len(set(retain)))))
		labels = np.array([d[x] if x in d else -1 for x in labels])

		if np.all(labels < 0):
			logging.warn("All cells were determined to be outliers!")
			return np.zeros_like(labels)

		if not self.outliers and np.any(labels == -1):
			# Assign each outlier to the same cluster as the nearest non-outlier
			nn = NNDescent(data=xy[labels >= 0], n_jobs=-1, random_state=0)
			nearest, _ = nn.query(xy[labels == -1], k=1)
			labels[labels == -1] = labels[labels >= 0][nearest.flat[:]]

		return labels
