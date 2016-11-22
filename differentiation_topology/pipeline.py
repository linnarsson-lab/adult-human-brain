import os
from datetime import datetime
import logging
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from palettable.tableau import Tableau_20
import loompy
import graph_tool.all as gt
import networkx as nx
import differentiation_topology as dt

def process_tissues(tissues, sample_dir="/Users/Sten/loom-datasets/Whole brain/", n_cores=4):
	"""
	Process a set of tissue, each comprising a set of raw Chromium samples

	Args:
		tissues (dict):	
	"""
	build_dir = "build_" + datetime.now().strftime("%Y%m%d_%H%M%S")
	os.mkdir(build_dir)

	work = [
		{
			"sample_dir": sample_dir,
			"build_dir": build_dir,
			"tissue": key,
			"samples": val,
		}
		for (key,val) in tissues.items()
	]

	pool = Pool(n_cores)
	pool.map(process_samples, work)

def process_samples(work, return_result=False):
	tissue = work["tissue"]
	samples = work["samples"]
	build_dir = work["build_dir"]
	sample_dir = work["sample_dir"]
	fname = os.path.join(build_dir, tissue.replace(" ", "_") + ".loom")

	# Since this will run in parallel processes, not threads, each log will go to the right place
	# (would not work if we used threads, since this configures the root logger)
	logging.basicConfig(filename=os.path.join(build_dir, tissue.replace(" ", "_") + ".log"))

	if tissue == "FAILED":
		logging.info("Skipping FAILED samples")
	logging.info("Processing: " + tissue)

	# Preprocessing
	logging.info("Preprocessing")
	(n_valid, n_total) = dt.preprocess(sample_dir, samples, fname, {"title": tissue}, False, False)
	logging.info("%d of %d cells were valid", n_valid, n_total)
	ds = loompy.connect(fname)

	# MKNN graph and initial clustering
	logging.info("Generating MKNN graph")
	(knn, genes, cells, sigma) = dt.knn_similarities(ds, cells=None, k=50, n_genes=1000, n_components = 100, min_cells=10, mutual=True, metric='euclidean', annoy_trees=50, filter_doublets=False)
	logging.info("Layout and Louvain-Jaccard clustering (1st round)")
	(g, labels, sfdp) = dt.make_graph(knn[cells, :][:, cells].tocoo(), jaccard=True)

	# Remove outliers
	logging.info("Removing outliers")
	g_pagerank = gt.pagerank(g).get_array()
	inliers = np.where(g_pagerank > np.percentile(g_pagerank, 2.5))[0]
	cells = cells[inliers]

	# Remove nodes with very long edges
	# logging.info("Removing stretched edges")
	# knn_filtered = knn[cells, :][:, cells]
	# v1 = sfdp[knn_filtered.row]
	# v2 = sfdp[knn_filtered.col]
	# edge_lengths = np.sqrt(np.power(v2[:, 0] - v1[:, 0], 2) + np.power(v2[:, 1] - v1[:, 1], 2))
	# mean_edge_length = np.zeros(knn_filtered.shape[0])
	# for ix in range(knn_filtered.shape[0]):
	# 	mean_edge_length[ix] = np.mean(edge_lengths[np.where(knn_filtered.row == ix)[0]])
	# inliers = np.where(mean_edge_length > np.percentile(mean_edge_length, 2.5))[0]
	# cells = cells[inliers]

	logging.info("Layout and Louvain-Jaccard clustering (2nd round)")
	(g, labels, sfdp) = dt.make_graph(knn[cells, :][:, cells].tocoo(), jaccard=True)

	# Plot the graph colored by Louvain cluster
	logging.info("Plotting")
	title = tissue + " (" + str(n_valid) + "/" + str(n_total) + " cells)"
	dt.plot_clusters(knn[:,cells][cells, :], labels, sfdp, title=title, outfile=os.path.join(build_dir, tissue.replace(" ", "_")))

	# Save the graph layout to the file
	logging.info("Saving attributes to the loom file")
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

	# Compute marker enrichment and trinarize
	logging.info("Marker enrichment and trinarization")
	(enrichment, trinary_prob, trinary_pat) = dt.expression_patterns(ds, labels, 0.05, 0.2, cells)
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
			f.write(("1" if (ds.row_attrs["_Excluded"][row] == 0 and ds.row_attrs["_Valid"][row] == 1) else "0") + "\t")
			for ix in range(enrichment.shape[1]):
				f.write(str(enrichment[row, ix]) + "\t")
			for ix in range(trinary_pat.shape[1]):
				f.write(str(trinary_pat[row, ix]) + "\t")
			for ix in range(trinary_prob.shape[1]):
				f.write(str(trinary_prob[row, ix]) + "\t")
			f.write("\n")
	ds.close()

	# Auto-annotation
	logging.info("Auto-annotating cell types and states")
	aa = dt.AutoAnnotator(ds)
	(tags, annotations) = aa.annotate(ds, trinary_prob)
	with open(os.path.join(build_dir, tissue.replace(" ", "_") + "_annotations.tab"), "w") as f:
		f.write("\t")
		for j in range(annotations.shape[1]):
			f.write("Cluster_" + str(j) + "\t")
		f.write("\n")
		for ix, tag in enumerate(tags):
			f.write(str(tag) + "\t")
			for j in range(annotations.shape[1]):
				f.write(str(annotations[ix, j])+"\t")
			f.write("\n")

	# Return the results of the last tissue, for debugging purposes (in case you want to replot etc)
	logging.info("Done")
	if return_result:
		return (knn, genes, cells, sigma, g, labels, sfdp, enrichment, trinary_prob, trinary_pat, tags, annotations)


def plot_clusters(knn, labels, sfdp, title=None, outfile=None):
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
		ax.text(x, y, str(lbl + 1), fontsize=9, bbox=dict(facecolor='gray', alpha=0.2, ec='none'))
	if outfile is not None:
		fig.savefig(outfile + "_clusters.png")
		plt.close()

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
		fig.savefig(outfile + "_mknn.png")
		plt.close()
