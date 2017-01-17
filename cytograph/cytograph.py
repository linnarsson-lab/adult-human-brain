import copy
import json
import logging
import os
import csv
from datetime import datetime
from typing import *
from multiprocessing import Pool
import loompy
import matplotlib.pyplot as plt
import numpy as np
from palettable.tableau import Tableau_20
from scipy import sparse
from scipy.special import polygamma
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from scipy.stats import ks_2samp
import networkx as nx
# import community
import cytograph as cg

colors20 = np.array(Tableau_20.mpl_colors)


# Run like this:
#
# import cytograph as cg
# c = cg.Cytograph("path_to_root")
# c.process("Cortex1")
#


class Cytograph:
	def __init__(self,
			root: str,
			sample_dir: str = "loom_samples",
			pool_config: str = "pooling_specification.tab",
			build_root: str = "loom_builds",
			annotation_root: str = "../auto-annotation",
			k: int = 30,
			lj_resolution: float = 1.0,
			n_genes: int = 2000,
			n_components: int = 50,
			pep: float = 0.05,
			f: float = 0.2,
			sfdp: bool = False
		) -> None:

		self.sample_dir = os.path.join(root, sample_dir)
		self.build_root = os.path.join(root, build_root)
		self.pool_config = os.path.join(root, pool_config)
		self.annotation_root = os.path.join(root, annotation_root)
		self.build_dir = None  # type: str
		self.k = k
		self.lj_resolution = lj_resolution
		self.n_components = n_components
		self.n_genes = n_genes
		self.pep = pep
		self.f = f
		self.plot_sfdp = sfdp

	def list_tissues(self) -> List[str]:
		temp = {}  # type: Dict[str, int]
		with open(self.pool_config, 'r') as f:
			for row in csv.reader(f, delimiter="\t"):
				if row[1] != "" and row[1] != "Pool":
					temp[row[1]] = 1
		return list(temp.keys())

	def process(self, tissues: List[str] = None, n_processes: int = 1) -> None:
		"""
		Process samples according to the pooling specification

		Args:
			tissues:	List of tissues to process, or None to process all
			n_processes: Number of parallel processes to use
		"""
		if isinstance(tissues, str):
			tissues = [tissues]
		if tissues is None:
			tissues = self.list_tissues()
		if not isinstance(tissues, (list, tuple)):
			raise ValueError("tissues must be a list of strings")

		self.build_dir = os.path.join(self.build_root, "build_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
		os.mkdir(self.build_dir)

		if len(tissues) > 1:
			pool = Pool(n_processes)
			pool.map(self._safe_proc_one, tissues)
		else:
			self._process_one(tissues[0])

	def _safe_proc_one(self, tissue: str) -> None:
		try:
			self._process_one(tissue)
		except Exception as e:
			logging.error(str(e))

	def _process_one(self, tissue: str) -> None:
		samples = []  # type: List[str]
		with open(self.pool_config, 'r') as f:
			for row in csv.reader(f, delimiter="\t"):
				if row[1] == tissue and row[2] != "FAILED":
					samples.append(row[0])
		if len(samples) == 0:
			logging.warn("No samples defined for " + tissue)
			return

		fname = os.path.join(self.build_dir, tissue.replace(" ", "_") + ".loom")

		# logging.basicConfig(filename=os.path.join(build_dir, tissue.replace(" ", "_") + ".log"))
		logging.info("Processing: " + tissue)

		# Preprocessing
		logging.info("Preprocessing " + tissue + " " + str(samples))
		cg.preprocess(self.sample_dir, self.build_dir, samples, fname, {"title": tissue}, False, True)

		ds = loompy.connect(fname)
		n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
		n_total = ds.shape[1]
		logging.info("%d of %d cells were valid", n_valid, n_total)
		logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])
		cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

		# logging.info("Facet learning")
		# labels = facets(ds, cells, config["facet_learning"])
		# logging.info(labels.shape)
		# n_labels = np.max(labels, axis=0) + 1
		# logging.info("Found " + str(n_labels) + " clusters")

		# KNN graph generation and clustering
		logging.info("Normalization and PCA projection")
		transformed = pca_projection(ds, cells, n_genes=self.n_genes, n_components=self.n_components)

		logging.info("Generating KNN graph")
		knn = kneighbors_graph(transformed, mode='distance', n_neighbors=self.k)
		knn = knn.tocoo()

		logging.info("Louvain-Jaccard clustering")
		lj = cg.LouvainJaccard(resolution=self.lj_resolution)
		labels = lj.fit_predict(knn)
		# g = lj.graph
		# Make labels for excluded cells == -1
		labels_all = np.zeros(ds.shape[1], dtype='int') + -1
		labels_all[cells] = labels

		# Mutual KNN
		mknn = knn.minimum(knn.transpose()).tocoo()

		# logging.info("t-SNE layout")
		# tsne_pos = TSNE(init=transformed[:, :2]).fit_transform(transformed)
		# # Place all cells in the lower left corner
		# tsne_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(tsne_pos, axis=0)
		# # Place the valid cells where they belong
		# tsne_all[cells] = tsne_pos

		if self.plot_sfdp:
			logging.info("SFDP layout")
			sfdp_pos = cg.SFDP().layout(lj.graph)
			sfdp_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(sfdp_pos, axis=0)
			sfdp_all[cells] = sfdp_pos

		logging.info("Marker enrichment and trinarization")
		(scores1, scores2, trinary_prob, trinary_pat) = cg.expression_patterns(ds, labels_all[cells], self.pep, self.f, cells)
		save_diff_expr(ds, self.build_dir, tissue, scores1*scores2, trinary_pat, trinary_prob)

		# Auto-annotation
		logging.info("Auto-annotating cell types and states")
		aa = cg.AutoAnnotator(ds, root=self.annotation_root)
		(tags, annotations) = aa.annotate(ds, trinary_prob)
		sizes = np.bincount(labels_all + 1)
		save_auto_annotation(self.build_dir, tissue, sizes, annotations, tags)

		logging.info("Plotting clusters on graph")
#		plot_clusters(mknn, labels, tsne_pos, tags, annotations, title=tissue, plt_labels=True, outfile=os.path.join(self.build_dir, tissue + "_tSNE"))
		plot_clusters(mknn, labels, transformed[:, :2], tags, annotations, title=tissue, plt_labels=True, outfile=os.path.join(self.build_dir, tissue + "_PCA"))
		if self.plot_sfdp:
			plot_clusters(mknn, labels, sfdp_pos, tags, annotations, title=tissue, plt_labels=True, outfile=os.path.join(self.build_dir, tissue + "_SFDP"))

		logging.info("Saving attributes")
#		ds.set_attr("_tSNE_X", tsne_all[:, 0], axis=1)
#		ds.set_attr("_tSNE_Y", tsne_all[:, 1], axis=1)
		if self.plot_sfdp:
			ds.set_attr("_SFDP_X", sfdp_all[:, 0], axis=1)
			ds.set_attr("_SFDP_Y", sfdp_all[:, 1], axis=1)
		ds.set_attr("Clusters", labels_all, axis=1)
		ds.set_edges("MKNN", cells[mknn.row], cells[mknn.col], mknn.data, axis=1)
		ds.set_edges("KNN", cells[knn.row], cells[knn.col], knn.data, axis=1)

		self.pca_transformed = transformed
#		self.tsne = tsne_all
		if self.plot_sfdp:
			self.sfdp = sfdp_all
		self.knn = knn
		self.mknn = mknn
		self.lj_graph = lj.graph
		self.labels = labels_all
		self.cells = cells
		self.aa_tags = tags
		self.aa_annotations = annotations
		self.enrichment = enrichment
		self.trinary_prob = trinary_prob
		logging.info("Done.")


def save_auto_annotation(build_dir: str, tissue: str, sizes: np.ndarray, annotations: np.ndarray, tags: np.ndarray) -> None:
	with open(os.path.join(build_dir, tissue.replace(" ", "_") + "_annotations.tab"), "w") as f:
		f.write("\t")
		for j in range(annotations.shape[1]):
			f.write(str(j + 1) + " (" + str(sizes[j]) + ")\t")
		f.write("\n")
		for ix, tag in enumerate(tags):
			f.write(str(tag) + "\t")
			for j in range(annotations.shape[1]):
				f.write(str(annotations[ix, j]) + "\t")
			f.write("\n")


def save_diff_expr(ds: loompy.LoomConnection, build_dir: str, tissue: str, enrichment: np.ndarray, trinary_pat: np.ndarray, trinary_prob: np.ndarray) -> None:
	with open(os.path.join(build_dir, tissue.replace(" ", "_") + "_diffexpr.tab"), "w") as f:
		f.write("Gene\t")
		f.write("Valid\t")
		for ix in range(enrichment.shape[1]):
			f.write("Enr_" + str(ix + 1) + "\t")
		for ix in range(trinary_pat.shape[1]):
			f.write("Trin_" + str(ix + 1) + "\t")
		for ix in range(trinary_prob.shape[1]):
			f.write("Prob_" + str(ix + 1) + "\t")
		f.write("\n")

		for row in range(enrichment.shape[0]):
			f.write(ds.Gene[row] + "\t")
			really_valid = 1
			if "_Valid" in ds.row_attrs and not ds.row_attrs["_Valid"][row] == 1:
				really_valid = 0
			if "_Excluded" in ds.row_attrs and not ds.row_attrs["_Excluded"][row] == 0:
				really_valid = 0
			f.write(str(really_valid) + "\t")
			for ix in range(enrichment.shape[1]):
				f.write(str(enrichment[row, ix]) + "\t")
			for ix in range(trinary_pat.shape[1]):
				f.write(str(trinary_pat[row, ix]) + "\t")
			for ix in range(trinary_prob.shape[1]):
				f.write(str(trinary_prob[row, ix]) + "\t")
			f.write("\n")


def facets(ds: loompy.LoomConnection, cells: np.ndarray, config: Dict) -> np.ndarray:
	"""
	Run Facet Learning on the dataset, with the given facets.
	"""
	n_genes = config["n_genes"]

	# Compute an initial gene set
	logging.info("Selecting genes for Facet Learning")
	with np.errstate(divide='ignore', invalid='ignore'):
		(genes, _, _) = feature_selection(ds, n_genes, cells)

	# Make sure the facet-specific genes are included in the gene set
	facet_genes = []  # type: List[str]
	for f in config["facets"]:
		facet_genes += f["genes"]
	facet_genes = np.where(np.in1d(ds.Gene, facet_genes))[0]
	genes = np.union1d(genes, facet_genes)

	logging.info("Loading data (in batches)")
	m = np.zeros((cells.shape[0], genes.shape[0]), dtype='int')
	j = 0
	for (_, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1, batch_size=5000):
		vals = vals[genes, :]
		vals = vals / (np.sum(vals, axis=0) + 1) * 10000
		vals = vals.transpose()
		n_cells_in_batch = selection.shape[0]
		m[j:j + n_cells_in_batch, :] = vals
		j += n_cells_in_batch

	logging.info("Facet learning with three facets")

	# Function to find indexes for a list of genes
	def gix(names: List[str]) -> List[int]:
		return [np.where(ds.Gene[genes] == n)[0][0] for n in names]

	facet_list = []  # type: List[cg.Facet]
	for f in config["facets"]:
		if "max_k" in f:
			max_k = f["max_k"]
		else:
			max_k = 0
		f0 = cg.Facet(f["name"], k=f["k"], n_genes=f["n_genes"], max_k=max_k, genes=gix(f["genes"]), adaptive=f["adaptive"])
		facet_list.append(f0)
	labels = cg.FacetLearning(facet_list, r=config["r"], max_iter=config["max_iter"], gene_names=ds.Gene[genes]).fit_transform(m)
	return labels


def prommt(ds: loompy.LoomConnection, cells: np.ndarray, config: Dict) -> np.ndarray:
	n_genes = config["n_genes"]
	n_S = config["n_S"]
	k = config["k"]
	max_iter = config["max_iter"]

	# Compute an initial gene set
	logging.info("Selecting genes for ProMMT")
	with np.errstate(divide='ignore', invalid='ignore'):
		(genes, _, _) = feature_selection(ds, n_genes, cells)

	logging.info("Loading data (in batches)")
	m = np.zeros((cells.shape[0], genes.shape[0]), dtype='int')
	j = 0
	for (_, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1, batch_size=5000):
		vals = vals[genes, :].transpose()
		n_cells_in_batch = selection.shape[0]
		m[j:j + n_cells_in_batch, :] = vals
		j += n_cells_in_batch

	logging.info("ProMMT clustering")
	labels = cg.ProMMT(n_S=n_S, k=k, max_iter=max_iter).fit_transform(m)
	return labels


class Normalizer(object):
	def __init__(self, ds: loompy.LoomConnection, standardize: bool = False, mu: np.ndarray = None, sd: np.ndarray = None) -> None:
		if (mu is None) or (sd is None):
			(self.sd, self.mu) = ds.map([np.std, np.mean], axis=0)
		else:
			self.sd = sd
			self.mu = mu
		self.totals = ds.map(np.sum, axis=1)
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
		n_genes:			Number of genes to use for PCA
		n_components:		Number of components to retain from PCA

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
	logging.info("Computing aggregate statistics for normalization")
	normalizer = Normalizer(ds, False, mu, sd)

	logging.info("Incremental PCA")
	pca = IncrementalPCA(n_components=n_components)
	for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1):
		vals = normalizer.normalize(vals, selection)
		pca.partial_fit(vals[genes, :].transpose())		# PCA on the selected genes

	logging.info("Projecting cells to PCA space (in batches)")
	transformed = np.zeros((cells.shape[0], pca.n_components_))
	j = 0
	for (_, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1):
		vals = normalizer.normalize(vals, selection)
		n_cells_in_batch = selection.shape[0]
		temp = pca.transform(vals[genes, :].transpose())
		transformed[j:j + n_cells_in_batch, :] = pca.transform(vals[genes, :].transpose())
		j += n_cells_in_batch

	pvalue_KS = np.zeros(transformed.shape[1])  # pvalue of each component
	for i in range(1, transformed.shape[1]):
		(_, pvalue_KS[i]) = ks_2samp(transformed[:, i - 1], transformed[:, i])
	sigs = np.where(pvalue_KS < 0.1)[0]
	if len(sigs) == 0:
		logging.info("No significant principal components!")
		sigs = (0,)
	logging.info("Using %d significant principal components", len(sigs))
	return transformed[:, sigs]


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


def plot_clusters(knn: np.ndarray, labels: np.ndarray, pos: Dict[int, Tuple[int, int]], tags: np.ndarray, annotations: np.ndarray, title: str = None, plt_labels: bool = True, outfile: str = None) -> None:
	# Plot auto-annotation
	fig = plt.figure(figsize=(10, 10))
	g = nx.from_scipy_sparse_matrix(knn)
	ax = fig.add_subplot(111)

	# Draw the KNN graph first, with gray transparent edges
	if title is not None:
		plt.title(title, fontsize=14, fontweight='bold')
	nx.draw_networkx_edges(g, pos=pos, alpha=0.1, width=0.1, edge_color='gray')

	# Then draw the nodes, colored by label
	block_colors = (np.array(Tableau_20.colors) / 255)[np.mod(labels, 20)]
	nx.draw_networkx_nodes(g, pos=pos, node_color=block_colors, node_size=10, alpha=0.5, linewidths=0)
	if plt_labels:
		for lbl in range(0, max(labels) + 1):
			if np.sum(labels == lbl) == 0:
				continue
			(x, y) = np.median(pos[np.where(labels == lbl)[0]], axis=0)
			text_labels = ["#" + str(lbl + 1)]
			for ix, a in enumerate(annotations[:, lbl]):
				if a >= 0.5:
					text_labels.append(tags[ix].abbreviation)
			text = "\n".join(text_labels)
			ax.text(x, y, text, fontsize=6, bbox=dict(facecolor='gray', alpha=0.3, ec='none'))
	if outfile is not None:
		fig.savefig(outfile + "_annotated.pdf")
		plt.close()
