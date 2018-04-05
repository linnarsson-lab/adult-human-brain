import numpy as np
import scipy.sparse as sparse
import networkx as nx
import community
import loompy
import cytograph as cg
from typing import *
import logging


def subspace_cosine_distance(ax: np.ndarray, bx: np.ndarray) -> np.ndarray:
	"""
	Calculate the weighted cosine similarity between two vectors living in (partially overlapping) subspaces

	Args:
		ax, bx		Input vectors, with positive values indicating the active subspace
	"""
	# Cosine similarities within the shared subspace
	subspace = ((ax >= 0) & (bx >= 0)).astype('int')
	a = subspace * ax
	b = subspace * bx
	s = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
	return s * subspace.sum() / ax.shape[0]
	
def _similarity(subspace: np.ndarray, ax: np.ndarray, bx: np.ndarray) -> np.ndarray:
	"""
	Calculate the weighted cosine similarity between two vectors living in (partially overlapping) subspaces
	"""
	# Cosine similarities within the shared subspace
	a = subspace * ax
	b = subspace * bx
	s = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
	return s * subspace.sum() / ax.shape[0]


def subspace_similarity_matrix(m: np.ndarray, subspaces: np.ndarray) -> np.ndarray:
	(n_samples, n_features) = m.shape
	result = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
		for j in range(n_samples):
			if j > i:
				break
			temp = _similarity(subspaces[i, :] * subspaces[j, :], m[i, :], m[j, :])
			result[i, j] = temp
			result[j, i] = temp
	return result


def subspace_knn_graph(m: np.ndarray, subspaces: np.ndarray, k: int = 100) -> np.ndarray:
	"""
	Return an adjacency matrix where entry (i, j) is 1 iff j is one of the k nearest neighbors of i
	"""
	(n_samples, n_features) = m.shape
	data = np.ones(k * n_samples, dtype='int32')
	rows = np.tile(np.arange(n_samples), k)  # 0 0 0 0 ... 1 1 1 1 1 .... 2 2 2 2 2 ...
	cols = np.zeros_like(rows)

	d = subspace_similarity_matrix(m, subspaces)
	for i in range(n_samples):
		cols[i * k: (i + 1) * k] = np.argsort(-d[i, :])[:k]
	return sparse.coo_matrix((data, (rows, cols)), shape=(n_samples, n_samples))


class ManifoldLearning4:
	def __init__(self, *, n_genes: int = 1000, k: int = 100) -> None:
		self.n_genes = n_genes
		self.k = k

	def fit(self, ds: loompy.LoomConnection) -> Tuple[sparse.coo_matrix, sparse.coo_matrix, np.ndarray]:
		"""
		Discover the manifold

		Returns:
			knn		The knn graph as a sparse matrix
			mknn	Mutual knn subgraph
			pos		2D projection (gt-SNE) as ndarray with shape (n_cells, 2)
		"""
		n_cells = ds.shape[1]
		logging.info("Processing all %d cells", n_cells)
		logging.info("Validating genes")
		nnz = ds.map([np.count_nonzero], axis=0)[0]
		valid_genes = np.logical_and(nnz > 5, nnz < ds.shape[1] * 0.5).astype("int")
		ds.ra._Valid = valid_genes
		logging.info("%d of %d genes were valid", np.sum(ds.ra._Valid == 1), ds.shape[0])

		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		logging.info("Selecting up to %d genes", self.n_genes)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)

		logging.info("Loading data for selected genes")
		data = np.zeros((n_cells, genes.shape[0]))
		for (ix, selection, view) in ds.scan(axis=1):
			data[selection - ix, :] = view[genes, :].T

		logging.info("Computing initial subspace KNN")
		subspaces = np.ones(data.shape)
		knn = subspace_knn_graph(data, subspaces)
		mknn = knn.minimum(knn.transpose()).tocoo()

		for t in range(5):
			logging.info(f"Refining subspace KNN (iteration {t + 1})")

			logging.info("Louvain clustering")
			graph = nx.from_scipy_sparse_matrix(mknn)
			partitions = community.best_partition(graph)
			labels = np.array([partitions[key] for key in range(mknn.shape[0])])
			ds.ca.Clusters = labels
			n_labels = np.max(labels) + 1
			logging.info(f"Found {n_labels} clusters")

			logging.info("Marker selection")
			(_, enrichment, _) = cg.MarkerSelection(n_markers=10, findq=False).fit(ds)
			subspaces = np.zeros(data.shape)
			for ix in range(enrichment.shape[1]):
				for j in range(n_cells):
					subspaces[j, np.argsort(-enrichment[:, ix])[:self.n_genes // n_labels]] = 1
			knn = subspace_knn_graph(data, subspaces)
			mknn = knn.minimum(knn.transpose()).tocoo()

		perplexity = min(self.k, (n_cells - 1) / 3 - 1)
		logging.info("gt-SNE layout")
		# Note that perplexity argument is ignored in this case, but must still be given
		# because bhtsne will check that it has a valid value
		tsne_pos = cg.TSNE(perplexity=perplexity).layout(data, knn=knn.tocsr())

		return (knn, mknn, tsne_pos)
