import numpy as np
import logging
import community
import networkx as nx
from scipy import sparse
from scipy.stats import mode
from typing import *
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
from scipy.stats import t, zscore


def grubbs(X: np.ndarray, test: str = 'two-tailed', alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
	'''
	Performs Grubbs' test for outliers recursively until the null hypothesis is
	true.

	Parameters
	----------
	X : ndarray
		A numpy array to be tested for outliers.
	test : str
		Describes the types of outliers to look for. Can be 'min' (look for
		small outliers), 'max' (look for large outliers), or 'two-tailed' (look
		for both).
	alpha : float
		The significance level.

	Returns
	-------
	X : ndarray
		A copy of the original array with outliers removed.
	outliers : ndarray
		An array of outliers.
	'''

	Z = zscore(X, ddof=1)  # Z-score
	N = len(X)  # number of samples

	# calculate extreme index and the critical t value based on the test
	if test == 'two-tailed':
		extreme_ix = lambda Z: np.abs(Z).argmax()
		t_crit = lambda N: t.isf(alpha / (2.*N), N-2)
	elif test == 'max':
		extreme_ix = lambda Z: Z.argmax()
		t_crit = lambda N: t.isf(alpha / N, N-2)
	elif test == 'min':
		extreme_ix = lambda Z: Z.argmin()
		t_crit = lambda N: t.isf(alpha / N, N-2)
	else:
		raise ValueError("Test must be 'min', 'max', or 'two-tailed'")

	# compute the threshold
	thresh = lambda N: (N - 1.) / np.sqrt(N) * \
		np.sqrt(t_crit(N)**2 / (N - 2 + t_crit(N)**2))

	# create array to store outliers
	outliers = np.array([])

	# loop throught the array and remove any outliers
	while abs(Z[extreme_ix(Z)]) > thresh(N):

		# update the outliers
		outliers = np.r_[outliers, X[extreme_ix(Z)]]
		# remove outlier from array
		X = np.delete(X, extreme_ix(Z))
		# repeat Z score
		Z = zscore(X, ddof=1)
		N = len(X)

	return X, outliers


class PolishedLouvain:
	def __init__(self, resolution: float = 1.0) -> None:
		self.resolution = resolution

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
		(_, outliers_x) = grubbs(x)
		(_, outliers_y) = grubbs(y)
		outliers = np.union1d(outliers_x, outliers_y)

		# See if the cluster is very dispersed
		min_pts = min(x.shape[0] - 1, max(5, round(0.1 * x.shape[0])))
		nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
		nn.fit(embedding)
		knn = nn.kneighbors_graph(mode='distance')
		k_radius = knn.max(axis=1).toarray()
		epsilon = np.percentile(k_radius, 70)

		# Not too many outliers, and not too dispersed
		if outliers.shape[0] <= 3 and (np.sqrt(x**2 + y**2) < epsilon).sum() >= min_pts:
			return np.zeros(embedding.shape[0], dtype='int')

		# Too many outliers, or too dispersed
		clusterer = DBSCAN(eps=epsilon, min_samples=min_pts)
		return clusterer.fit_predict(embedding)
		
	def fit_predict(self, knn: sparse.coo_matrix, embedding: np.ndarray) -> np.ndarray:
		"""
		Given a sparse adjacency matrix, perform Louvain clustering, then polish the result

		Args:
			knn:		The sparse adjacency matrix
			embedding: 	The 2D embedding of the graph, shape (n_cells, 2)

		Returns:
			labels:	The cluster labels (where -1 indicates outliers)

		"""
		logging.info("Louvain community detection")
		g = nx.from_scipy_sparse_matrix(knn)
		partitions = community.best_partition(g, resolution=self.resolution, randomize=False)
		labels = np.array([partitions[key] for key in range(knn.shape[0])])

		# Mark tiny clusters as outliers
		logging.info("Marking tiny clusters as outliers")
		bigs = np.where(np.bincount(labels) >= 10)[0]
		mapping = {k: v for v, k in enumerate(bigs)}
		labels = np.array([mapping[x] if x in bigs else -1 for x in labels])

		# Mark outliers using DBSCAN
		logging.info("Using DBSCAN to mark outliers")
		nn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree", n_jobs=4)
		nn.fit(embedding)
		knn = nn.kneighbors_graph(mode='distance')
		k_radius = knn.max(axis=1).toarray()
		epsilon = np.percentile(k_radius, 80)
		clusterer = DBSCAN(eps=epsilon, min_samples=10)
		outliers = (clusterer.fit_predict(embedding) == -1)
		labels[outliers] = -1

		# Mark outliers as cells in bad neighborhoods
		logging.info("Using neighborhood to mark outliers")
		nn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree", n_jobs=4)
		nn.fit(embedding)
		knn = nn.kneighbors_graph(mode='connectivity').tocoo()
		temp = []
		for ix in range(labels.shape[0]):
			if labels[ix] == -1:
				temp.append(-1)
				continue
			neighbors = knn.col[np.where(knn.row == ix)[0]]
			neighborhood = labels[neighbors] == labels[ix]
			if neighborhood.sum() / neighborhood.shape[0] > 0.2:
				temp.append(labels[ix])
			else:
				temp.append(-1)

		# Renumber the clusters
		retain = sorted(list(set(labels)))
		logging.info(str(retain))
		d = dict(zip(retain, np.arange(-1, len(set(retain)))))
		labels = np.array([d[x] if x in d else -1 for x in labels])
		logging.info(str(sorted(list(set(labels)))))

		# Break clusters based on the embedding
		logging.info("Breaking clusters")
		max_label = 0
		labels2 = np.copy(labels)
		for lbl in range(labels.max() + 1):
			cluster = labels == lbl
			adjusted = self._break_cluster(embedding[cluster, :])
			new_labels = np.copy(adjusted)
			for i in range(np.max(adjusted) + 1):
				new_labels[adjusted == i] = i + max_label
			max_label = max_label + np.max(adjusted) + 1
			labels2[cluster] = new_labels
		labels = labels2
		logging.info(str(sorted(list(set(labels)))))

		# Set the local cluster label to the local majority vote
		logging.info("Smoothing cluster identity on the embedding")
		nn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree", n_jobs=4)
		nn.fit(embedding)
		knn = nn.kneighbors_graph(mode='connectivity').tocoo()
		temp = []
		for ix in range(labels.shape[0]):
			if labels[ix] == -1:
				temp.append(-1)
				continue
			neighbors = knn.col[np.where(knn.row == ix)[0]]
			temp.append(mode(labels[neighbors])[0][0])

		# Renumber the clusters (since some clusters might have been lost in poor neighborhoods)
		retain = sorted(list(set(labels)))
		logging.info(str(retain))
		d = dict(zip(retain, np.arange(-1, len(set(retain)))))
		labels = np.array([d[x] if x in d else -1 for x in labels])
		logging.info((sorted(list(set(labels)))))

		return labels
