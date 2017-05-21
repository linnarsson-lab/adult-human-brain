from typing import *
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
import numpy as np
import logging
import cytograph as cg
import loompy


class ManifoldLearning:
	def __init__(self, n_genes: int = 1000, gtsne: bool = True) -> None:
		self.n_genes = n_genes
		self.gtsne = gtsne

	def fit(self, ds: loompy.LoomConnection) -> Tuple[sparse.coo_matrix, sparse.coo_matrix, np.ndarray]:
		"""
		Discover the manifold

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

		logging.info("Selecting %d genes", self.n_genes)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		temp = np.zeros(ds.shape[0])
		temp[genes] = 1
		ds.set_attr("_Selected", temp, axis=0)

		logging.info("PCA projection")
		pca = cg.PCAProjection(genes, max_n_components=50)
		pca_transformed = pca.fit_transform(ds, normalizer, cells=cells)
		transformed = pca_transformed

		logging.info("Generating KNN graph")
		k = 10
		nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=4)
		nn.fit(transformed)
		knn = nn.kneighbors_graph(mode='connectivity')
		knn = knn.tocoo()
		mknn = knn.minimum(knn.transpose()).tocoo()

		logging.info("Louvain-Jaccard clustering")
		lj = cg.LouvainJaccard(resolution=1)
		labels = lj.fit_predict(knn)
		# Make labels for excluded cells == -1
		labels_all = np.zeros(ds.shape[1], dtype='int') + -1
		labels_all[cells] = labels
		ds.set_attr("Clusters", labels_all, axis=1)
		n_labels = np.max(labels) + 1
		logging.info("Found " + str(n_labels) + " LJ clusters")

		logging.info("Marker selection")
		(genes, _) = cg.MarkerSelection(n_markers=int(500 / n_labels)).fit(ds)

		logging.info("PCA projection")
		pca = cg.PCAProjection(genes, max_n_components=50)
		# Select cells across clusters more uniformly, preventing a single cluster from dominating the PCA
		cells_adjusted = cg.cap_select(labels, cells, int(n_valid * 0.2))
		pca.fit(ds, normalizer, cells=cells_adjusted)
		# Note that here we're transforming all cells; we just did the fit on the selection
		transformed = pca.transform(ds, normalizer, cells=cells)

		logging.info("Generating multiscale KNN graph")
		k = 100
		nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=4)
		nn.fit(transformed)
		knn = nn.kneighbors(return_distance=False)  # shape: (n_cells, k)
		n_cells = knn.shape[0]
		a = np.tile(np.arange(n_cells), k)
		b = np.reshape(knn.T, (n_cells * k,))
		w = np.repeat(1 / np.arange(1, k + 1), n_cells)
		knn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells))
		threshold = w > 0.05
		mknn = sparse.coo_matrix((w[threshold], (a[threshold], b[threshold])), shape=(n_cells, n_cells))
		mknn = mknn.minimum(mknn.transpose()).tocoo()

		if self.gtsne:
			logging.info("gt-SNE layout")
			tsne_pos = cg.TSNE().layout(transformed, knn=knn.tocsr())
		else:
			logging.info("t-SNE layout")
			tsne_pos = cg.TSNE(perplexity=k).layout(transformed)
		tsne_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(tsne_pos, axis=0)
		tsne_all[cells] = tsne_pos

		# Transform back to the full set of cells
		knn = sparse.coo_matrix((knn.data, (cells[knn.row], cells[knn.col])), shape=(n_total, n_total))
		mknn = sparse.coo_matrix((mknn.data, (cells[mknn.row], cells[mknn.col])), shape=(n_total, n_total))

		return (knn, mknn, tsne_all)
