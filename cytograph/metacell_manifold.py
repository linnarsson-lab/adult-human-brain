from typing import *
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
import scipy.sparse as sparse
import numpy as np
import logging
import cytograph as cg
import loompy
import igraph


class NegativeBinomialTest:
	def __init__(self, alpha: float, phi: float, epsilon: int = 1) -> None:
		self.alpha = alpha
		self.phi = phi
		self.epsilon = epsilon
		
	def test(self, a: np.ndarray, b: np.ndarray) -> float:
		alpha_corr = 10**(np.log(self.alpha)/1000)
		if self.phi == 0:
			intervals = np.array(stats.poisson.interval(alpha_corr, a + self.epsilon)).T
		else:
			r = 1 - self.phi
			intervals = np.array(stats.nbinom.interval(alpha_corr, (a + self.epsilon) / (1 / r**2 - 1), r**2)).T
		n_out = np.sum(np.logical_or(b < intervals[:, 0], b > intervals[:, 1]))
		return n_out == 0


class MetacellManifold:
	def __init__(self, *, n_genes: int = 1000, gtsne: bool = True, alpha: float = 1, genes: np.ndarray = None, filter_cellcycle: str = None, layer: str=None) -> None:
		self.n_genes = n_genes
		self.gtsne = gtsne
		self.alpha = alpha
		self.genes = genes
		self.filter_cellcycle = filter_cellcycle
		self.layer = layer

	def fit(self, ds: loompy.LoomConnection) -> Tuple[sparse.coo_matrix, sparse.coo_matrix, np.ndarray]:
		"""
		Discover the manifold
		Args:
			n_genes		Number of genes to use for manifold learning (ignored if genes is not None)
			gtsnse		Use graph t-SNE for layout (default: standard tSNE)
			alpha		The scale parameter for multiscale KNN
			genes		List of genes to use for manifold learning

		Returns:
			knn		The multiscale knn graph as a sparse matrix, with k = 100
			mknn	Mutual knn subgraph, with k = 20
			pos		2D projection (t-SNE or gt-SNE) as ndarray with shape (n_cells, 2)

		Remarks:
			Input must already be filtered to remove bad cells; all cells are assumed valid
		"""
		n_total = ds.shape[1]

		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)

		if self.filter_cellcycle is not None:
			cell_cycle_genes = np.array(open(self.filter_cellcycle).read().split())
			mask = np.in1d(ds.Gene, cell_cycle_genes)
			if np.sum(mask) == 0:
				logging.warn("None cell cycle genes where filtered, check your gene list")
		else:
			mask = None

		if self.genes is None:
			logging.info("Selecting up to %d genes", self.n_genes)
			genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd, mask=mask)

			n_components = min(50, n_valid)
			logging.info("PCA projection to %d components", n_components)
			pca = cg.PCAProjection(genes, max_n_components=n_components)
			pca_transformed = pca.fit_transform(ds, normalizer, cells=cells)
			transformed = pca_transformed

			logging.info("Generating balanced KNN graph")
			k = min(100, n_valid - 1)
			bnn = cg.BalancedKNN(k=k)
			bnn.fit(transformed)
			knn = bnn.kneighbors_graph(mode='connectivity')
			knn = knn.tocoo()
			mknn = knn.minimum(knn.transpose()).tocoo()

			logging.info("MKNN-Louvain clustering with outliers")
			(a, b, w) = (mknn.row, mknn.col, mknn.data)
			G = igraph.Graph(list(zip(a, b)), directed=False, edge_attrs={'weight': w})
			VxCl = G.community_multilevel(return_levels=False, weights="weight")
			labels = np.array(VxCl.membership)
			bigs = np.where(np.bincount(labels) >= 10)[0]
			mapping = {k: v for v, k in enumerate(bigs)}
			labels = np.array([mapping[x] if x in bigs else -1 for x in labels])

			# Make labels for excluded cells == -1
			labels_all = np.zeros(ds.shape[1], dtype='int') + -1
			labels_all[cells] = labels
			ds.set_attr("Clusters", labels_all, axis=1)
			n_labels = np.max(labels) + 1
			logging.info("Found " + str(n_labels) + " clusters")

			logging.info("Marker selection")
			(genes, _, _) = cg.MarkerSelection(n_markers=int(500 / n_labels)).fit(ds)
		else:
			genes = self.genes

		temp = np.zeros(ds.shape[0])
		temp[genes] = 1
		ds.set_attr("_Selected", temp, axis=0)
		logging.info("%d genes selected", temp.sum())

		if self.genes is None:
			# Select cells across clusters more uniformly, preventing a single cluster from dominating the PCA
			cells_adjusted = cg.cap_select(labels - labels.min(), cells, int(n_valid * 0.2))
			n_components = min(50, cells_adjusted.shape[0])
			logging.info("PCA projection to %d components", n_components)
			pca = cg.PCAProjection(genes, max_n_components=n_components)
			pca.fit(ds, normalizer, cells=cells_adjusted)
		else:
			n_components = min(50, cells.shape[0])
			logging.info("PCA projection to %d components", n_components)
			pca = cg.PCAProjection(genes, max_n_components=n_components)
			pca.fit(ds, normalizer, cells=cells)
			
		# Note that here we're transforming all cells; we just did the fit on the selection
		transformed = pca.transform(ds, normalizer, cells=cells)

		k = min(100, n_valid - 1)
		logging.info("Generating multiscale KNN graph (k = %d)", k)
		bnn = cg.BalancedKNN(k=k)
		bnn.fit(transformed)
		knn = bnn.kneighbors(mode='connectivity')[1][:, 1:]
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
			# Note that perplexity argument is ignored in this case, but must still be given
			# because bhtsne will check that it has a valid value
			tsne_pos = cg.TSNE(perplexity=perplexity).layout(transformed, knn=knn.tocsr())
		else:
			logging.info("t-SNE layout")
			tsne_pos = cg.TSNE(perplexity=perplexity).layout(transformed)
		tsne_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(tsne_pos, axis=0)
		tsne_all[cells] = tsne_pos

		# Transform back to the full set of cells
		knn = sparse.coo_matrix((knn.data, (cells[knn.row], cells[knn.col])), shape=(n_total, n_total))
		mknn = sparse.coo_matrix((mknn.data, (cells[mknn.row], cells[mknn.col])), shape=(n_total, n_total))

		return (knn, mknn, tsne_all)
