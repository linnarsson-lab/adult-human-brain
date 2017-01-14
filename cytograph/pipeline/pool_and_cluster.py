import os
from typing import *
import logging
from shutil import copyfile
import numpy as np
import loompy
import differentiation_topology as dt
import luigi
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
from differentiation_topology.pipeline import QualityControl, AutoAnnotation
from sklearn.svm import SVR


class PoolAndCluster(luigi.Task):
	"""
	Luigi Task
		- pool a set of individual Loom files
		- validate genes across the pooled cells
		- perform clustering
	"""
	build_folder = luigi.Parameter(default="")
	in_files = luigi.ListParameter()
	out_file = luigi.Parameter()
	n_genes = luigi.IntParameter(default=5000)
	n_components = luigi.IntParameter(default=50)

	def output(self) -> luigi.LocalTarget:
		return luigi.LocalTarget(os.path.join(self.build_folder, self.out_file))

	def requires(self) -> List[Any]:
		if self.tags is None:
			return [QualityControl(sample=f[:-5]) for f in self.in_files]
		else:
			return [AutoAnnotation(name=f[:-5]) for f in self.in_files]

	def run(self) -> None:
		logging.info("Joining files for clustering as " + self.name)
		fname = os.path.join(self.build_folder, self.out_file)
		loompy.join([os.path.join(self.build_folder, f) for f in self.in_files], fname, key="Accession", file_attrs={})

		# Validate genes
		ds = loompy.connect(fname)
		nnz = ds.map(np.count_nonzero, axis=0)
		valid = np.logical_and(nnz > 20, nnz < ds.shape[1] * 0.6)
		ds.set_attr("_Valid", valid, axis=0)

		n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
		n_total = ds.shape[1]
		logging.info("%d of %d cells were valid", n_valid, n_total)
		logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])
		cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

		# KNN graph generation and clustering
		logging.info("Normalization and PCA projection")
		transformed = pca_projection(ds, cells, self.n_genes, self.n_components)

		logging.info("Generating KNN graph")
		knn = kneighbors_graph(transformed, mode='distance', n_neighbors=30)
		knn = knn.tocoo()

		logging.info("Louvain-Jaccard clustering")
		lj = dt.LouvainJaccard(resolution=1.0)
		labels = lj.fit_predict(knn)
		g = lj.graph
		# Make labels for excluded cells == -1
		labels_all = np.zeros(ds.shape[1], dtype='int') + -1
		labels_all[cells] = labels

		# Mutual KNN
		mknn = knn.minimum(knn.transpose()).tocoo()

		logging.info("t-SNE layout")
		tsne_pos = TSNE(init=transformed[:, :2], perplexity=50).fit_transform(transformed)
		tsne_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(tsne_pos, axis=0)
		tsne_all[cells] = tsne_pos

		logging.info("Saving attributes")
		ds.set_attr("_tSNE_X", tsne_all[:, 0], axis=1)
		ds.set_attr("_tSNE_Y", tsne_all[:, 1], axis=1)
		ds.set_attr("Clusters", labels_all, axis=1)
		ds.set_edges("MKNN", cells[mknn.row], cells[mknn.col], mknn.data)
		ds.set_edges("KNN", cells[knn.row], cells[knn.col], knn.data)
		ds.close()

class Normalizer(object):
	def __init__(self, ds: loompy.LoomConnection, rescale: bool = True, standardize: bool = False, mu: np.ndarray = None, sd: np.ndarray = None) -> None:
		if (mu is None) or (sd is None):
			(self.sd, self.mu) = ds.map([np.std, np.mean], axis=0)
		else:
			self.sd = sd
			self.mu = mu
		self.totals = ds.map(np.sum, axis=1)
		self.rescale = rescale
		self.standardize = standardize

	def normalize(self, vals: np.ndarray, cells: np.ndarray) -> np.ndarray:
		"""
		Normalize a matrix using the previously calculated aggregate statistics

		Args:
			vals (ndarray):		Matrix of shape (n_genes, n_cells)
			cells (ndarray):	Indices of the cells that are represented in vals

		Returns:
			vals_adjusted (ndarray):	The normalized values
		"""
		if self.rescale:
			# Adjust total count per cell to 10,000
			vals = vals / (self.totals[cells] + 1) * 10000
		# Log transform
		vals = np.log(vals + 1)
		# Subtract mean per gene
		vals = vals - self.mu[:, None]
		if self.standardize:
			# Scale to unit standard deviation per gene
			vals = self._div0(vals, self.sd[:, None])
		return vals

	def _div0(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
		""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
		with np.errstate(divide='ignore', invalid='ignore'):
			c = np.true_divide(a, b)
			c[~np.isfinite(c)] = 0  # -inf inf NaN
		return c


def pca_projection(ds: loompy.LoomConnection, cells: np.ndarray, n_genes: int, n_components: int) -> np.ndarray:
	"""
	Memory-efficient PCA projection of the dataset

	Args:
		ds (LoomConnection): 	Dataset
		cells (ndaray of int):	Indices of cells to project
		n_genes: 				Number of genes to select for PCA
		n_components:			Numnber of PCs to retain

	Returns:
		The dataset transformed by the top principal components
		Shape: (n_samples, n_components), where n_samples = cells.shape[0]
	"""
	n_cells = cells.shape[0]

	# Compute an initial gene set
	logging.info("Selecting genes")
	with np.errstate(divide='ignore', invalid='ignore'):
		(genes, mu, sd) = feature_selection(ds, n_genes, cells)

	# Perform PCA based on the gene selection and the cell sample
	normalizer = Normalizer(ds, True, False, mu, sd)

	logging.info("Computing PCA incrementally")
	pca = IncrementalPCA(n_components=n_components)
	for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1):
		vals = normalizer.normalize(vals, selection)
		pca.partial_fit(vals[genes, :].transpose())		# PCA on the selected genes

	logging.info("Projecting cells to PCA space")
	transformed = np.zeros((cells.shape[0], pca.n_components_))
	j = 0
	for (_, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1):
		vals = normalizer.normalize(vals, selection)
		n_cells_in_batch = selection.shape[0]
		temp = pca.transform(vals[genes, :].transpose())
		transformed[j:j + n_cells_in_batch, :] = pca.transform(vals[genes, :].transpose())
		j += n_cells_in_batch

	return transformed


def feature_selection(ds: loompy.LoomConnection, n_genes: int, cells: np.ndarray = None, cache: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Fits a noise model (CV vs mean)

	Args:
		ds (LoomConnection):	Dataset
		n_genes (int):	number of genes to include
		cells (ndarray): cells to include when computing mean and CV (or None)
		cache (ndarray): dataset corresponding to the selected cells (or None)

	Returns:
		ndarray of selected genes (list of ints)
	"""
	(mu, std) = ds.map((np.mean, np.std), axis=0, selection=cells)

	valid = np.logical_and(
		np.logical_and(
			ds.row_attrs["_Valid"] == 1,
			ds.row_attrs["Gene"] != "Xist"
		),
		ds.row_attrs["Gene"] != "Tsix"
	).astype('int')

	ok = np.logical_and(mu > 0, std > 0)
	cv = std[ok] / mu[ok]
	log2_m = np.log2(mu[ok])
	log2_cv = np.log2(cv)

	svr_gamma = 1000. / len(mu[ok])
	clf = SVR(gamma=svr_gamma)
	clf.fit(log2_m[:, np.newaxis], log2_cv)
	fitted_fun = clf.predict
	# Score is the relative position with respect of the fitted curve
	score = log2_cv - fitted_fun(log2_m[:, np.newaxis])
	score = score * valid[ok]
	top_genes = np.where(ok)[0][np.argsort(score)][-n_genes:]

	logging.debug("Keeping %i genes" % top_genes.shape[0])
	# logging.info(str(sorted(ds.Gene[top_genes[:50]])))
	return (top_genes, mu, std)
