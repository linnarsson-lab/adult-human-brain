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
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
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
			sfdp: bool = False,
			auto_annotate: bool = True
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
		self.auto_annotate = auto_annotate

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
		self.cells = cells

		# logging.info("Facet learning")
		# labels = facets(ds, cells, config["facet_learning"])
		# logging.info(labels.shape)
		# n_labels = np.max(labels, axis=0) + 1
		# logging.info("Found " + str(n_labels) + " clusters")

		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)

		logging.info("Feature selection")
		genes = cg.FeatureSelection(self.n_genes).fit(ds)
		temp = np.zeros(ds.shape[0])
		temp[genes] = 1
		ds.set_attr("_Selected", temp, axis=0)
		plot_cv_mean(genes, normalizer.mu, normalizer.sd, os.path.join(self.build_dir, tissue))

		logging.info("PCA projection")
		self.pca = cg.PCAProjection(genes, max_n_components=self.n_components)
		pca_transformed = self.pca.fit_transform(ds, normalizer, cells=cells)
		self.pca_transformed = pca_transformed

		logging.info("FastICA projection")
		self.ica = FastICA()
		ica_transformed = self.ica.fit_transform(pca_transformed)

		# Use ICA
		transformed = ica_transformed

		logging.info("Generating KNN graph")
		knn = kneighbors_graph(transformed, mode='distance', n_neighbors=self.k)
		knn = knn.tocoo()
		self.knn = knn

		logging.info("Louvain-Jaccard clustering")
		lj = cg.LouvainJaccard(resolution=self.lj_resolution)
		labels = lj.fit_predict(knn)
		# g = lj.graph
		# Make labels for excluded cells == -1
		labels_all = np.zeros(ds.shape[1], dtype='int') + -1
		labels_all[cells] = labels
		self.lj_graph = lj.graph
		self.labels = labels_all

		# Mutual KNN
		mknn = knn.minimum(knn.transpose()).tocoo()
		self.mknn = mknn

		logging.info("t-SNE layout")
		tsne_pos = TSNE(init=transformed[:, :2]).fit_transform(transformed)
		# Place all cells in the lower left corner
		tsne_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(tsne_pos, axis=0)
		# Place the valid cells where they belong
		tsne_all[cells] = tsne_pos
		self.tsne = tsne_all

		if self.plot_sfdp:
			logging.info("SFDP layout")
			sfdp_pos = cg.SFDP().layout(lj.graph)
			sfdp_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(sfdp_pos, axis=0)
			sfdp_all[cells] = sfdp_pos
			self.sfdp = sfdp_all

		logging.info("Marker enrichment and trinarization")
		(scores1, scores2, trinary_prob, trinary_pat) = cg.expression_patterns(ds, labels_all[cells], self.pep, self.f, cells)
		save_diff_expr(ds, self.build_dir, tissue, scores1 * scores2, trinary_pat, trinary_prob)
		self.enrichment = scores1 * scores2
		self.trinary_prob = trinary_prob

		# Auto-annotation
		tags = None  # type: np.ndarray
		annotations = None  # type: np.ndarray
		if self.auto_annotate:
			logging.info("Auto-annotating cell types and states")
			aa = cg.AutoAnnotator(ds, root=self.annotation_root)
			(tags, annotations) = aa.annotate(ds, trinary_prob)
			sizes = np.bincount(labels_all + 1)
			save_auto_annotation(self.build_dir, tissue, sizes, annotations, tags)
			self.aa_tags = tags
			self.aa_annotations = annotations

		logging.info("Plotting clusters on graph")
		plot_clusters(mknn, labels, tsne_pos, tags, annotations, title=tissue, outfile=os.path.join(self.build_dir, tissue + "_tSNE"))
		plot_clusters(mknn, labels, pca_transformed[:, :2], tags, annotations, title=tissue, outfile=os.path.join(self.build_dir, tissue + "_PCA"))
		plot_clusters(mknn, labels, ica_transformed[:, :2], tags, annotations, title=tissue, outfile=os.path.join(self.build_dir, tissue + "_ICA"))
		if self.plot_sfdp:
			plot_clusters(mknn, labels, sfdp_pos, tags, annotations, title=tissue, outfile=os.path.join(self.build_dir, tissue + "_SFDP"))

		logging.info("Saving attributes")
		ds.set_attr("_tSNE_X", tsne_all[:, 0], axis=1)
		ds.set_attr("_tSNE_Y", tsne_all[:, 1], axis=1)
		if self.plot_sfdp:
			ds.set_attr("_SFDP_X", sfdp_all[:, 0], axis=1)
			ds.set_attr("_SFDP_Y", sfdp_all[:, 1], axis=1)
		ds.set_attr("Clusters", labels_all, axis=1)
		ds.set_edges("MKNN", cells[mknn.row], cells[mknn.col], mknn.data, axis=1)
		ds.set_edges("KNN", cells[knn.row], cells[knn.col], knn.data, axis=1)

		logging.info("Done.")


def plot_cv_mean(genes: np.ndarray, mu: np.ndarray, std: np.ndarray, outfile: str) -> None:
	x1 = mu[genes]
	y1 = std[genes] / mu[genes]
	fig = plt.figure(figsize=(10, 6))
	ax1 = fig.add_subplot(111)
	ax1.scatter(np.log(mu), np.log(std / mu), c='grey', marker=".", edgecolors="none")
	ax1.scatter(np.log(x1), np.log(y1), c='blue', marker=".", edgecolors="none")
	if outfile is not None:
		fig.savefig(outfile + "_gene_selection.pdf")
		plt.close()


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


def plot_clusters(knn: np.ndarray, labels: np.ndarray, pos: Dict[int, Tuple[int, int]], tags: np.ndarray, annotations: np.ndarray, title: str = None, outfile: str = None) -> None:
	# Plot auto-annotation
	fig = plt.figure(figsize=(10, 10))
	g = nx.from_scipy_sparse_matrix(knn)
	ax = fig.add_subplot(111)
	plt_labels = True if tags is not None else False

	# Draw the KNN graph first, with gray transparent edges
	if title is not None:
		plt.title(title, fontsize=14, fontweight='bold')
	nx.draw_networkx_edges(g, pos=pos, alpha=0.1, width=0.1, edge_color='gray')

	# Then draw the nodes, colored by label
	block_colors = (np.array(Tableau_20.colors) / 255)[np.mod(labels, 20)]
	nx.draw_networkx_nodes(g, pos=pos, node_color=block_colors, node_size=10, alpha=0.5, linewidths=0)
	for lbl in range(0, max(labels) + 1):
		if np.sum(labels == lbl) == 0:
			continue
		(x, y) = np.median(pos[np.where(labels == lbl)[0]], axis=0)
		text_labels = ["#" + str(lbl + 1)]
		if plt_labels:
			for ix, a in enumerate(annotations[:, lbl]):
				if a >= 0.5:
					text_labels.append(tags[ix].abbreviation)
		text = "\n".join(text_labels)
		ax.text(x, y, text, fontsize=6, bbox=dict(facecolor='gray', alpha=0.3, ec='none'))
	if outfile is not None:
		fig.savefig(outfile + "_annotated.pdf")
		plt.close()
