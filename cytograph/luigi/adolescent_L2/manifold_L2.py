from typing import *
import os
from shutil import copyfile
import numpy as np
import logging
import luigi
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


def cap_select(labels: np.ndarray, items: np.ndarray, max_n: int) -> np.ndarray:
	"""
	Return a list of items but with no more than max_n entries
	having each unique label
	"""
	n_labels = np.max(labels) + 1
	sizes = np.bincount(labels, minlength=n_labels)
	result = []  # type: List[int]
	for lbl in range(n_labels):
		n = min(max_n, sizes[lbl])
		selected = np.where(labels == lbl)[0]
		result = result + list(np.random.choice(selected, n, False))
	return items[np.array(result)]


class ManifoldL2(luigi.Task):
	"""
	Luigi Task to learn the high-dimensional manifold and embed it as a multiscale KNN graph, as well as t-SNE projection
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	n_genes = luigi.IntParameter(default=1000)
	gtsne = luigi.BoolParameter(default=True)

	def requires(self) -> luigi.Task:
		return cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".manifold.txt"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)

			n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
			n_total = ds.shape[1]
			logging.info("%d of %d cells were valid", n_valid, n_total)
			logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])
			cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

			logging.info("Normalization")
			normalizer = cg.Normalizer(False)
			normalizer.fit(ds)
			with np.errstate(divide='ignore', invalid='ignore'):
				ds.set_attr("_LogCV", np.log(normalizer.sd) - np.log(normalizer.mu))
				ds.set_attr("_LogMean", np.log(normalizer.mu))

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
			ds.set_edges("KNN", cells[knn.row], cells[knn.col], knn.data, axis=1)
			mknn = knn.minimum(knn.transpose()).tocoo()
			ds.set_edges("MKNN", cells[mknn.row], cells[mknn.col], mknn.data, axis=1)

			logging.info("Louvain-Jaccard clustering")
			lj = cg.LouvainJaccard(resolution=10)
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
			cells_adjusted = cap_select(labels, cells, int(n_valid * 0.2))
			pca.fit(ds, normalizer, cells=cells_adjusted)
			# Note that here we're transforming all cells; we just did the fit on the selection
			pca_transformed = pca.transform(ds, normalizer, cells=cells)
			transformed = pca_transformed

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
			ds.set_edges("KNN", cells[knn.row], cells[knn.col], knn.data, axis=1)
			threshold = w > 0.05
			mknn = sparse.coo_matrix((w[threshold], (a[threshold], b[threshold])), shape=(n_cells, n_cells))
			mknn = mknn.minimum(mknn.transpose()).tocoo()
			ds.set_edges("MKNN", cells[mknn.row], cells[mknn.col], mknn.data, axis=1)

			if self.gtsne:
				logging.info("gt-SNE layout")
				tsne_pos = cg.TSNE().layout(transformed, knn=knn.tocsr())
			else:
				logging.info("t-SNE layout")
				tsne_pos = cg.TSNE(perplexity=k).layout(transformed)
			tsne_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(tsne_pos, axis=0)
			tsne_all[cells] = tsne_pos
			ds.set_attr("_X", tsne_all[:, 0], axis=1)
			ds.set_attr("_Y", tsne_all[:, 1], axis=1)
			with open(out_file, "w") as f:
				f.write(str(n_labels) + " LJ clusters\n")
			ds.close()
