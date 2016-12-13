import os
from datetime import datetime
import logging
import copy
import json
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from palettable.tableau import Tableau_20
import loompy
import graph_tool.all as gt
import networkx as nx
import differentiation_topology as dt


config = {
	"sample_dir": "/Users/Sten/loom-datasets/Whole brain/",
	"build_dir": None,
	"entrypoint": "preprocessing",
	"tissue": "Dentate gyrus",
	"samples": ["10X43_1", "10X46_1"],
	
	"preprocessing": {
		"do_validate_genes": True,
		"make_doublets": False
	},

	"graph": {
		"cells": None,
		"n_genes": 1000,
		"n_components": 100,
		"normalize": False,
		"standardize": True,
		"metric": 'euclidean',
		"mutual": True,
		"k": 50,
		"kth": 5,
		"use_radius": True,
		"min_cells": 10,
		"n_trees": 50,
		"search_k_factor": 10,
		"filter_doublets": False,
		"outlier_percentile": 2.5,
		"edge_percentile": 0.5,
		"louvain_resolution": 1,
		"edge_weights": "jaccard", # or "euclidean"
		"cooling_step": 0.95
	},
	"radius_graph": {
		"cells": None,
		"n_genes": 1000,
		"n_components": 100,
		"normalize": False,
		"standardize": True,
		"min_cells": 10,
		"outlier_percentile": 2.5,
		"edge_percentile": 0.5,
		"louvain_resolution": 1,
		"edge_weights": "jaccard", # or "euclidean"
		"cooling_step": 0.95
	},
	"annotation": {
		"pep": 0.05,
		"f": 0.2,
		"annotation_root": "/Users/Sten/Dropbox (Linnarsson Group)/Code/autoannotation/"
	}
}

def get_default_config():
	return copy.deepcopy(config)


def process_many(configs, n_cores):
	"""
	Process a set of raw Chromium samples

	Args:
		configs (list):
	"""
	for ix, config in enumerate(configs):
		if config["build_dir"] is None:
			if config["build_dir"] is None:
				config["build_dir"] = "build_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(ix)
				os.mkdir(config["build_dir"])

	pool = Pool(n_cores)
	pool.map(process_one, configs)

def process_one(config, return_result=False):
	if config["build_dir"] is None:
		config["build_dir"] = "build_" + datetime.now().strftime("%Y%m%d_%H%M%S")
		os.mkdir(config["build_dir"])
	build_dir = config["build_dir"]
	tissue = config["tissue"]
	samples = config["samples"]
	sample_dir = config["sample_dir"]
	fname = os.path.join(build_dir, tissue.replace(" ", "_") + ".loom")
	stage = config["entrypoint"]

	with open(os.path.join(build_dir, "build_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"), 'w') as f:
		f.write(json.dumps(config))

	# Since this will run in parallel processes, not threads, each log will go to the right place
	# (would not work if we used threads, since this configures the root logger)
	logging.basicConfig(filename=os.path.join(build_dir, tissue.replace(" ", "_") + ".log"))

	logging.info("Processing: " + tissue)

	# Preprocessing
	if stage == None or stage == "preprocessing":
		stage = None
		logging.info("Preprocessing")
		dt.preprocess(sample_dir, samples, fname, {"title": tissue}, False, False)
	ds = loompy.connect(fname)
	n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
	n_total = ds.shape[1]
	logging.info("%d of %d cells were valid", n_valid, n_total)

	# MKNN graph and initial clustering
	if stage == None or stage == "graph":
		stage = None
		logging.info("Generating MKNN graph")
		(knn, genes, cells, sigma) = dt.knn_similarities(ds, config["graph"])
		logging.info("Layout and Louvain-Jaccard clustering (1st round)")
		(g, labels, sfdp) = dt.make_graph(knn[cells, :][:, cells].tocoo(), jaccard=True)

		# Remove outliers
		logging.info("Removing outliers")
		g_pagerank = gt.pagerank(g).get_array()
		inliers = np.where(g_pagerank > np.percentile(g_pagerank, config["graph"]["outlier_percentile"]))[0]

		# Remove nodes with very long edges
		logging.info("Removing stretched edges")
		knn_filtered = knn[cells, :][:, cells].tocoo()
		v1 = sfdp[knn_filtered.row]
		v2 = sfdp[knn_filtered.col]
		edge_lengths = np.sqrt(np.power(v2[:, 0] - v1[:, 0], 2) + np.power(v2[:, 1] - v1[:, 1], 2))
		mean_edge_length = np.zeros(knn_filtered.shape[0])
		for ix in range(knn_filtered.shape[0]):
			mean_edge_length[ix] = np.mean(edge_lengths[np.where(knn_filtered.row == ix)[0]])
		inliers2 = np.where(mean_edge_length > np.percentile(mean_edge_length, config["graph"]["edge_percentile"]))[0]

		# Remove very small L-J clusters
		sizes = np.bincount(labels)
		inliers3 = np.where((sizes > config["graph"]["min_cells"])[labels])[0]

		# Collect all the ok cells
		cells = cells[np.intersect1d(np.intersect1d(inliers, inliers2), inliers3)]

		logging.info("Layout and Louvain-Jaccard clustering (2nd round)")
		(g, labels, sfdp) = dt.make_graph(knn[cells, :][:, cells].tocoo(), jaccard=True, cooling_step=config["graph"]["cooling_step"])

		# Save the graph layout to the file
		logging.info("Saving attributes to the loom file")
		valids = np.zeros(ds.shape[1])
		valids[cells] = 1
		ds.set_attr("_Valid", valids, axis=1, dtype="int")
		x = np.zeros(ds.shape[1])
		x[cells] = sfdp[:, 0]
		y = np.zeros(ds.shape[1])
		y[cells] = sfdp[:, 1]
		ds.set_attr("SFDP_X", x, axis=1, dtype="float64")
		ds.set_attr("SFDP_Y", y, axis=1, dtype="float64")
		# Save the LJ cluster labels
		lj = np.zeros(ds.shape[1])
		lj[cells] = labels + 1 # Make sure clusters start at 1 (zero reserved for non-valid cells)
		ds.set_attr("Louvain_Jaccard", lj, axis=1, dtype="int")

		# Save graph to build dir
		g.save(os.path.join(build_dir, tissue + ".graphml.gz"))
	else:
		# Load (g, labels, sfdp) from the build_dir 
		g = gt.load_graph(os.path.join(build_dir, tissue + ".graphml.gz"))
		labels = ds.col_attrs["Louvain_Jaccard"]
		x = ds.col_attrs["SFDP_X"]
		y = ds.col_attrs["SFDP_Y"]
		sfdp = np.vstack((x,y)).transpose()
		cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

	# Compute marker enrichment and trinarize
	# MKNN graph and initial clustering
	if stage == None or stage == "annotation":
		stage = None
		logging.info("Marker enrichment and trinarization")
		(enrichment, trinary_prob, trinary_pat) = dt.expression_patterns(ds, labels, config["annotation"]["pep"], config["annotation"]["f"], cells)
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
				if "_Valid" in ds.row_attrs:
					f.write(("1" if (ds.row_attrs["_Excluded"][row] == 0 and ds.row_attrs["_Valid"][row] == 1) else "0") + "\t")
				else:
					f.write(("1" if (ds.row_attrs["_Excluded"][row] == 0) else "0") + "\t")
				for ix in range(enrichment.shape[1]):
					f.write(str(enrichment[row, ix]) + "\t")
				for ix in range(trinary_pat.shape[1]):
					f.write(str(trinary_pat[row, ix]) + "\t")
				for ix in range(trinary_prob.shape[1]):
					f.write(str(trinary_prob[row, ix]) + "\t")
				f.write("\n")

		# Auto-annotation
		logging.info("Auto-annotating cell types and states")
		aa = dt.AutoAnnotator(ds)
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

		# Plot the graph colored by Louvain cluster
		logging.info("Plotting")
		title = tissue + " (" + str(cells.shape[0]) + "/" + str(n_total) + " cells)"
		dt.plot_clusters(knn[:,cells][cells, :], labels, sfdp, tags, annotations, title=title, outfile=os.path.join(build_dir, tissue.replace(" ", "_")))

	# Return the results of the last tissue, for debugging purposes (in case you want to replot etc)
	logging.info("Done")
	if return_result:
		return (knn, genes, cells, sigma, g, labels, sfdp, enrichment, trinary_prob, trinary_pat, tags, annotations)


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
