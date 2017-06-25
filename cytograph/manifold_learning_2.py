from typing import *
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
from sklearn.decomposition import NMF
import numpy as np
import logging
import cytograph as cg
import loompy


class ManifoldLearning2:
	def __init__(self, n_genes: int = 5000, gtsne: bool = True, alpha: float = 1) -> None:
		self.n_genes = n_genes
		self.gtsne = gtsne
		self.alpha = alpha

	def fit(self, ds: loompy.LoomConnection) -> Tuple[sparse.coo_matrix, sparse.coo_matrix, np.ndarray]:
		"""
		Discover the manifold using non-negative matrix factorization

		Returns:
			knn		The multiscale knn graph as a sparse matrix, with k = 100
			mknn	Mutual knn subgraph, with k = 20
			pos		2D projection (t-SNE or gt-SNE) as ndarray with shape (n_cells, 2)
		"""
		n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
		n_total = ds.shape[1]
		logging.info("%d of %d cells were valid", n_valid, n_total)
		logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])
		cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)

		logging.info("Selecting up to %d genes", self.n_genes)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		temp = np.zeros(ds.shape[0])
		temp[genes] = 1
		ds.set_attr("_Selected", temp, axis=0)
		logging.info("%d genes selected", temp.sum())

		n_components = min(100, cells.shape[0])
		logging.info("NMF projection to %d components", n_components)
		nmf = NMF(n_components=n_components, init='nndsvd')
		transformed = nmf.fit_transform(ds[:, :][genes, :][:, cells].T)

		k = min(100, n_valid - 1)
		logging.info("Generating multiscale KNN graph (k = %d)", k)
		nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=4, metric='euclidean')
		nn.fit(transformed)
		knn = nn.kneighbors(return_distance=False)  # shape: (n_cells, k)
		n_cells = knn.shape[0]
		a = np.tile(np.arange(n_cells), k)
		b = np.reshape(knn.T, (n_cells * k,))
		w = np.repeat(1 / np.power(np.arange(1, k + 1), self.alpha), n_cells)
		knn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells))
		threshold = w > 0.05
		mknn = sparse.coo_matrix((w[threshold], (a[threshold], b[threshold])), shape=(n_cells, n_cells))
		mknn = mknn.minimum(mknn.transpose()).tocoo()

		perplexity = min(k, (n_valid - 1) / 3 - 1)
		if self.gtsne:
			logging.info("gt-SNE layout")
			tsne_pos = cg.TSNE(perplexity=perplexity, n_dims=3).layout(transformed, knn=knn.tocsr())
		else:
			logging.info("t-SNE layout")
			tsne_pos = cg.TSNE(perplexity=perplexity, n_dims=3).layout(transformed)
		tsne_all = np.zeros((ds.shape[1], 3), dtype='int') + np.min(tsne_pos, axis=0)
		tsne_all[cells] = tsne_pos

		# Transform back to the full set of cells
		knn = sparse.coo_matrix((knn.data, (cells[knn.row], cells[knn.col])), shape=(n_total, n_total))
		mknn = sparse.coo_matrix((mknn.data, (cells[mknn.row], cells[mknn.col])), shape=(n_total, n_total))

		return (knn, mknn, tsne_all)
