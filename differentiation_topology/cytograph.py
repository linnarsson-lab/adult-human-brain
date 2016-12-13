import copy
import json
import logging
import os
from datetime import datetime
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
import graph_tool.all as gt
from annoy import AnnoyIndex
import networkx as nx
import community
import differentiation_topology as dt

colors20 = np.array(Tableau_20.mpl_colors)

default_config = {
	"sample_dir": "/Users/Sten/loom-datasets/Whole brain/",
	"build_dir": None,
	"tissue": "Dentate gyrus",
	"samples": ["10X43_1", "10X46_1"],
	
	"preprocessing": {
		"do_validate_genes": True,
		"make_doublets": False
	},
	"cytograph": {
		"cache_n_columns": 5000,
		"n_components": 100,
		"min_components": 2,
		"min_cells": 10,
		"k": 50,
		"n_trees": 50,
		"n_genes": 1000,
		"normalize": True,
		"standardize": True,
		"trinarize_f": 0.2,
        "trinarize_pep": 0.05,
		"min_diff_genes": 5,
		"level1_n_clusters": 10
	},
	"annotation": {
		"pep": 0.05,
		"f": 0.2,
		"annotation_root": "/Users/Sten/Dropbox (Linnarsson Group)/Code/autoannotation/"
	}
}

def get_default_config():
	return copy.deepcopy(default_config)

class TreeNode(object):
	def __init__(self, cells, children):
		self.cells = cells
		self.children = children

	def has_children(self):
		return not (self.children is None)

	def get_labels(self):
		return np.array(sorted(self.get_labels_dict().items()))[:, 1]

	def get_labels_dict(self):
		"""
		Return labels for the cells in this node, as a dict mapping cells to labels
		"""
		if not self.has_children():
			return dict(zip(self.cells, np.zeros(self.cells.shape[0], dtype='int')))
		left_labels = self.children[0].get_labels_dict().copy()
		for i in range(1, len(self.children)):
			offset = max(left_labels.values()) + 1
			right_labels = self.children[i].get_labels_dict()
			for key, val in right_labels.items():
				left_labels[key] = val + offset
		return left_labels

# class TreeNode(object):
# 	def __init__(self, cells, left, right):
# 		self.cells = cells
# 		self.right = right
# 		self.left = left
# #		self.vector = vector

# 	def has_children(self):
# 		return not (self.left is None and self.right is None)

# 	def get_labels(self):
# 		return np.array(sorted(self.get_labels_dict().items()))[:, 1]

# 	def get_labels_dict(self):
# 		"""
# 		Return labels for the cells in this node, as a dict mapping cells to labels
# 		"""
# 		if not self.has_children():
# 			return dict(zip(self.cells, np.zeros(self.cells.shape[0], dtype='int')))
# 		left_labels = self.left.get_labels_dict().copy()
# 		offset = max(left_labels.values()) + 1
# 		right_labels = self.right.get_labels_dict()
# 		for key, val in right_labels.items():
# 			left_labels[key] = val + offset
# 		return left_labels

	# def get_cluster_vectors(self):
	# 	return np.vstack(self._vectors())

	# def _accumulate_vectors(self):
	# 	if self.has_children():
	# 		return self._accumulate_vectors(self.left) + self._accumulate_vectors(self.right)
	# 	else:
	# 		return [self.vector]


def cytograph(config):
	skip_preprocessing = False
	if config["build_dir"] is None:
		config["build_dir"] = "build_" + datetime.now().strftime("%Y%m%d_%H%M%S")
		os.mkdir(config["build_dir"])
	else:
		skip_preprocessing = True
	build_dir = config["build_dir"]
	tissue = config["tissue"]
	samples = config["samples"]
	sample_dir = config["sample_dir"]
	fname = os.path.join(build_dir, tissue.replace(" ", "_") + ".loom")

	config_json = "build_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
	with open(os.path.join(build_dir, config_json), 'w') as f:
		f.write(json.dumps(config))

	#logging.basicConfig(filename=os.path.join(build_dir, tissue.replace(" ", "_") + ".log"))
	logging.info("Processing: " + tissue)

	# Preprocessing
	if not skip_preprocessing:
		logging.info("Preprocessing")
		dt.preprocess(sample_dir, samples, fname, {"title": tissue}, False, True)

	ds = loompy.connect(fname)
	n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
	n_total = ds.shape[1]
	logging.info("%d of %d cells were valid", n_valid, n_total)
	logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])

	# logging.info("Generating MKNN graph")
	# (knn, genes, cells, sigma) = dt.knn_similarities(ds, config["graph"])
	# logging.info("Layout and Louvain-Jaccard clustering")
	# (g, lj_labels, sfdp) = dt.make_graph(knn[cells, :][:, cells].tocoo(), jaccard=True)

	# Tree splitting
	cells = np.where(ds.col_attrs["_Valid"] == 1)[0]
	(tree, knn) = tree_split(ds, cells, config["cytograph"], top_level=True)
	labels = tree.get_labels()
	n_labels = max(labels)+1
	logging.info("Found %d clusters", n_labels)

	# Enrichment
	logging.info("Marker enrichment and trinarization")
	f = config["annotation"]["f"]
	pep = config["annotation"]["pep"]
	(enrichment, trinary_prob, trinary_pat) = dt.expression_patterns(ds, labels, pep, f, cells)
	with open(os.path.join(build_dir, tissue.replace(" ", "_") + "_diffexpr.tab"), "w") as f:
		f.write("Gene\t")
		f.write("Valid\t")
		for ix in range(enrichment.shape[1]):
			f.write("Enr_" + str(ix+1) + "\t")
		for ix in range(trinary_pat.shape[1]):
			f.write("Trin_" + str(ix+1) + "\t")
		for ix in range(trinary_prob.shape[1]):
			f.write("Prob_" + str(ix+1) + "\t")
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

	# Auto-annotation
	logging.info("Auto-annotating cell types and states")
	aa = dt.AutoAnnotator(ds, root=config["annotation"]["annotation_root"])
	(tags, annotations) = aa.annotate(ds, trinary_prob)
	sizes = np.bincount(labels)
	with open(os.path.join(build_dir, tissue.replace(" ", "_") + "_annotations.tab"), "w") as f:
		f.write("\t")
		for j in range(annotations.shape[1]):
			f.write(str(j + 1) + " (" + str(sizes[j]) + ")\t")
		f.write("\n")
		for ix, tag in enumerate(tags):
			f.write(str(tag) + "\t")
			for j in range(annotations.shape[1]):
				f.write(str(annotations[ix, j])+"\t")
			f.write("\n")

	logging.info("Done.")
	return (tree, knn, tags, annotations, enrichment, trinary_pat, trinary_prob)

class Normalizer(object):
	def __init__(self, ds, config, mu=None, sd=None):
		if (mu is None) or (sd is None):
			(self.sd, self.mu) = ds.map([np.std, np.mean], axis=0)
		else:
			self.sd = sd
			self.mu = mu
		self.totals = ds.map(np.sum, axis=1)
		self.config = config

	def normalize(self, vals, cells):
		"""
		Normalize a matrix using the previously calculated aggregate statistics

		Args:
			vals (ndarray):		Matrix of shape (n_genes, n_cells)
			cells (ndarray):	Indices of the cells that are represented in vals

		Returns:
			vals_adjusted (ndarray):	The normalized values		
		"""
		if self.config["normalize"]:
			# Adjust total count per cell to 10,000
			vals = vals/(self.totals[cells]+1)*10000
		# Log transform
		vals = np.log(vals + 1)
		# Subtract mean per gene
		vals = vals - self.mu[:, None]
		if self.config["standardize"]:
			# Scale to unit standard deviation per gene
			vals = self._div0(vals, self.sd[:, None])
		return vals

	def _div0(self, a, b):
		""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
		with np.errstate(divide='ignore', invalid='ignore'):
			c = np.true_divide(a, b)
			c[~np.isfinite(c)] = 0  # -inf inf NaN
		return c

def tree_split(ds, cells, config, branch="0", cache=None, n_genes_override=None, top_level=False):
	"""
	Split the dataset in two parts recursively

	Args:
		ds (LoomConnection):	Dataset
		cells (ndarray):		List of selected cells (indices of columns in ds)
		config (dict):			Config parameters
		branch (str):			Current branch in the tree (for logging only)
		cache (ndarray):		Cached dataset corresponding to the selected cells

	Returns:
		tree (TreeNode):		A TreeNode object describing the split
	"""
	cache_n_columns = config["cache_n_columns"]
	n_genes = config["n_genes"]
	n_trees = config["n_trees"]
	if not (n_genes_override is None):
		n_genes = n_genes_override
	n_components = config["n_components"]
	if cache is not None:
		n_components = min(min(n_components, cache.shape[1]), n_genes)
	min_components = config["min_components"]
	min_cells = config["min_cells"]
	trinarize_f = config["trinarize_f"]
	trinarize_pep = config["trinarize_pep"]
	min_diff_genes = config["min_diff_genes"]

	logging.info("At branch %s with %d cells", branch, cells.shape[0])

	# if cache is not None:
	# 	logging.info("Cache: " + str(cache.shape))

	# Compute an initial gene set
	logging.info("Selecting genes")
	with np.errstate(divide='ignore', invalid='ignore'):
		(genes, mu, sd) = feature_selection(ds, n_genes, cells, cache)

	# Perform PCA based on the gene selection and the cell sample
#	logging.info("Computing %d PCA components", n_components)

	if top_level:
		logging.info("Computing aggregate statistics for normalization")
		# This should be done at the top level only (to ensure aggregate stats are valid across entire dataset)
		normalizer = Normalizer(ds, config, mu, sd)

		logging.info("Incremental PCA in batches of %d", cache_n_columns)
		pca = IncrementalPCA(n_components=n_components)
		for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1, batch_size=cache_n_columns):
			vals = normalizer.normalize(vals, ix + selection)
			pca.partial_fit(vals[genes, :].transpose())		# PCA on the selected genes

		logging.info("Projecting cells to PCA space (in batches)")
		transformed = np.zeros((cells.shape[0], pca.n_components_))
		j = 0
		for (_, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=1, batch_size=cache_n_columns):
			vals = normalizer.normalize(vals, selection)
			n_cells_in_batch = selection.shape[0]
			temp = pca.transform(vals[genes, :].transpose())
			transformed[j:j + n_cells_in_batch, :] = pca.transform(vals[genes, :].transpose())
			j += n_cells_in_batch

		(knn, ok_cells, sigma) = make_knn(transformed, ds.shape[1], cells, k=config["k"])
	else:
		logging.info("PCA using cached values")
		pca = PCA(n_components=n_components)
		vals = cache[genes, :].transpose()
		transformed = pca.fit_transform(vals)

	bs = dt.broken_stick(len(genes), n_components)
	sig = pca.explained_variance_ratio_ > bs
	n_sig = np.sum(sig)
	logging.info("Found %d significant principal components (but using all %d)", n_sig, n_components)
	if n_sig < min_components:
		logging.info("Not splitting, because not enough principal components")
		return TreeNode(cells, None)

	if top_level:
		actual_n_clusters = config["level1_n_clusters"]
		logging.info("Computing KNN graph")
		aggr = AgglomerativeClustering(linkage='ward', connectivity=knn[cells, :][:, cells], n_clusters=actual_n_clusters)
		logging.info("KNN-constrained agglomerative clustering")
		labels = aggr.fit_predict(transformed)
	else:
		actual_n_clusters = 2
		logging.info("Split by KMeans")
		labels = KMeans(n_clusters=2).fit_predict(transformed)

	logging.info("Found clusters with %s cells", np.bincount(labels))

	# Illustrate the split
	tsne = TSNE().fit_transform(transformed)
	plt.scatter(tsne[:, 0], tsne[:, 1], c=colors20[labels], lw=0.1)
	plt.title(branch)
	plt.savefig(branch + "_tsne.png")
	plt.close()

	plt.scatter(transformed[:, 0], transformed[:, 1], c=colors20[labels], lw=0.1)
	plt.title(branch)
	plt.savefig(branch + "_pca.png")
	plt.close()

	# Determine if this cluster should be split or not
	if not top_level:
		# Check that we're not making a very small cluster
		n_left = np.sum(labels == 0)
		n_right = np.sum(labels == 1)
		if n_left < min_cells or n_right < min_cells:
			logging.info("Not splitting, because one cluster would be too small")
			return TreeNode(cells, None)

		# Check if there are enough differentially expressed genes
		if cache is not None:
			logging.info("Trinarizing and checking for differentially expressed genes")
			diffs = 0
			for gene in genes:
				(_, trinary) = dt.betabinomial_trinarize_array(cache[gene, :], labels, trinarize_pep, trinarize_f)
				if (trinary[0] == 0 and trinary[1] == 1) or (trinary[0] == 1 and trinary[1] == 0):
					diffs += 1
					if diffs >= min_diff_genes:
						break
			logging.info("Found %d discordant genes", diffs)
			if not diffs >= min_diff_genes:
				logging.info("Not splitting, because %d < %d discordant genes", diffs, min_diff_genes)
				return TreeNode(cells, None)

	trees = []
	for i in range(max(labels) + 1):
		if top_level:
			logging.info("Loading cells into cache")
			vals = ds[:, cells[labels == i]]
			child_cache = normalizer.normalize(vals, cells[labels == i])
		else:
			child_cache = cache[:, labels == i]
		trees.append(tree_split(ds, cells[labels == i], config, branch + str(i), cache=child_cache))
	if top_level:
		return (TreeNode(cells, trees), knn)
	else:
		return TreeNode(cells, trees)

def make_knn(m, d, cells, n_trees=50, k=50, kth=5, min_cells=10, use_radius=False, mutual=True):
	n_components = m.shape[1]
	logging.info("Creating approximate nearest neighbors model (annoy)")
	annoy = AnnoyIndex(n_components, metric="euclidean")
	for ix, cell in enumerate(cells):
		annoy.add_item(cell, m[ix, :])
	annoy.build(n_trees)

	logging.info("Computing mutual nearest neighbors")
	I = np.empty(d*k)
	J = np.empty(d*k)
	V = np.empty(d*k)
	sigma = np.empty((d,), dtype='float64') # The local kernel width
	for i in range(d):
		(nn, w) = annoy.get_nns_by_item(i, k, include_distances=True)
		w = np.array(w)
		I[i*k:(i+1)*k] = [i]*k
		J[i*k:(i+1)*k] = nn
		V[i*k:(i+1)*k] = w
		sigma[i] = w[kth-1]

	if use_radius:
		# Compute a radius neighborhood based on the average local kernel width (sigma)
		radius = np.mean(sigma)
		logging.info("Mean distance to kth neighbor: " + str(radius))
		logging.info("Computing radius neighbors")
		ball_tree = BallTree(m)
		(J, V) = ball_tree.query_radius(m, r=radius, return_distance=True)
		I = []
		for ix, rn in enumerate(J):
			I.extend([ix]*rn.shape[0])
		J = np.concatenate(J).ravel()
		V = np.concatenate(V).ravel()
		I = np.array(I)

	# k nearest neighbours
	knn = sparse.coo_matrix((V, (I, J)), shape=(d, d))

	data = knn.data
	rows = knn.row
	cols = knn.col

	# Convert to similarities by rescaling and subtracting from 1
	data = data / data.max()
	data = 1 - data
	sigma = sigma / sigma.max()
	sigma = 1 - sigma

	knn = sparse.coo_matrix((data, (rows, cols)), shape=(d, d)).tocsr()

	if mutual:
		# Compute Mutual knn
		# This removes all edges that are not reciprocal
		knn = knn.minimum(knn.transpose())
	else:
		# Make all edges reciprocal
		# This duplicates all edges that are not reciprocal
		knn = knn.maximum(knn.transpose())

	# Find and remove disconnected components
	logging.info("Identifying cells in small components")
	(_, labels) = sparse.csgraph.connected_components(knn, directed='False')
	sizes = np.bincount(labels)
	ok_cells = np.where((sizes > min_cells)[labels])[0]
	logging.info("Small components contained %d cells", cells.shape[0] - ok_cells.shape[0])

	return (knn, ok_cells, sigma)

def feature_selection(ds, n_genes, cells=None, cache=None):
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
	if cache is None:
		(mu, std) = ds.map((np.mean, np.std), axis=0, selection=cells)
	else:
		mu = cache.mean(axis=1)
		std = cache.std(axis=1)

	valid = np.logical_and(
				np.logical_and(
					ds.row_attrs["_Valid"] == 1, 
					ds.row_attrs["Gene"] != "Xist"
					),
				ds.row_attrs["Gene"] != "Tsix"
			).astype('int')

	ok = np.logical_and(mu > 0, std > 0)
	cv = std[ok]/mu[ok]
	log2_m = np.log2(mu[ok])
	log2_cv = np.log2(cv)

	svr_gamma = 1000./len(mu[ok])
	clf = SVR(gamma=svr_gamma)
	clf.fit(log2_m[:, np.newaxis], log2_cv)
	fitted_fun = clf.predict
	# Score is the relative position with respect of the fitted curve
	score = log2_cv - fitted_fun(log2_m[:, np.newaxis])
	score = score*valid[ok]
	top_genes = np.where(ok)[0][np.argsort(score)][-n_genes:]

	logging.debug("Keeping %i genes" % top_genes.shape[0])
	logging.info(sorted(ds.Gene[top_genes[:50]]))
	return (top_genes, mu, std)


def plot_clusters(knn, labels, sfdp, tags, annotations, title=None, outfile=None):
	# Plot auto-annotation
	fig = plt.figure(figsize=(10, 10))
	block_colors = (np.array(Tableau_20.colors)/255)[np.mod(labels, 20)]
	ax = fig.add_subplot(111)
	if title is not None:
		plt.title(title, fontsize=14, fontweight='bold')
	nx.draw(
		nx.from_scipy_sparse_matrix(knn),
		pos = sfdp, 
		node_color=block_colors,
		node_size=10,
		alpha=0.5, 
		width=0.1, 
		linewidths=0, 
		cmap='prism'
	)
	for lbl in range(max(labels)):
		(x, y) = sfdp[np.where(labels == lbl)[0]].mean(axis=0)
		text_labels = []
		for ix,a in enumerate(annotations[:,lbl]):
			if a >= 0.5:
				text_labels.append(tags[ix].abbreviation)
		if len(text_labels) > 0:
			text = "\n".join(text_labels)
		else:
			text = str(lbl + 1)
		ax.text(x, y, text, fontsize=6, bbox=dict(facecolor='gray', alpha=0.2, ec='none'))
	if outfile is not None:
		fig.savefig(outfile + "_annotated.pdf")
		plt.close()

	# Plot cluster labels
	fig = plt.figure(figsize=(10, 10))
	block_colors = (np.array(Tableau_20.colors)/255)[np.mod(labels, 20)]
	ax = fig.add_subplot(111)
	if title is not None:
		plt.title(title, fontsize=14, fontweight='bold')
	nx.draw(
		nx.from_scipy_sparse_matrix(knn), 
		pos = sfdp, 
		node_color=block_colors, 
		node_size=10, 
		alpha=0.5, 
		width=0.1, 
		linewidths=0, 
		cmap='prism'
	)
	for lbl in range(max(labels)):
		(x, y) = sfdp[np.where(labels == lbl)[0]].mean(axis=0)
		ax.text(x, y, str(lbl + 1), fontsize=6, bbox=dict(facecolor='gray', alpha=0.2, ec='none'))
	if outfile is not None:
		fig.savefig(outfile + "_clusters.pdf")
		plt.close()

	# Plot MKNN edges only
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)
	if title is not None:
		plt.title(title, fontsize=14, fontweight='bold')
	nx.draw(
		nx.from_scipy_sparse_matrix(knn),
		pos = sfdp,
		width=0.1,
		linewidths=0.1,
		nodelist=[]
	)
	for lbl in range(max(labels)):
		(x, y) = sfdp[np.where(labels == lbl)[0]].mean(axis=0)
		ax.text(x, y, str(lbl + 1), fontsize=9, bbox=dict(facecolor='gray', alpha=0.2, ec='none'))
	if outfile is not None:
		fig.savefig(outfile + "_mknn.pdf")
		plt.close()
