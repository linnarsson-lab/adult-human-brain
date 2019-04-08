from typing import *
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import math
import networkx as nx
import cytograph as cg
import loompy
from matplotlib.colors import LinearSegmentedColormap
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
import community
from .utils import species
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.spatial import ConvexHull
from .midpoint_normalize import MidpointNormalize


def mad(points, thresh=2.5):
	"""
	Returns a boolean array with True if points are outliers and False 
	otherwise.

	Parameters:
	-----------
		points : An numobservations by numdimensions array of observations
		thresh : The modified z-score to use as a threshold. Observations with
			a modified z-score (based on the median absolute deviation) greater
			than this value will be classified as outliers.

	Returns:
	--------
		mask : A numobservations-length boolean array.

	References:
	----------
		Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
		Handle Outliers", The ASQC Basic References in Quality Control:
		Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
	"""
	if len(points.shape) == 1:
		points = points[:,None]
	median = np.median(points, axis=0)
	diff = np.sum((points - median)**2, axis=-1)
	diff = np.sqrt(diff)
	med_abs_deviation = np.median(diff)

	modified_z_score = 0.6745 * diff / med_abs_deviation

	return modified_z_score > thresh


def TFs(ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, layer: str = "pooled", out_file_root: str = None) -> None:
	TFs = cg.TFs_human if species(ds) == "Homo sapiens" else cg.TFs_mouse
	enrichment = dsagg["enrichment"][:, :]
	enrichment = enrichment[np.isin(dsagg.ra.Gene, TFs), :]
	genes = dsagg.ra.Gene[np.isin(dsagg.ra.Gene, TFs)]
	genes = genes[np.argsort(-enrichment, axis=0)[:10, :]].T  # (n_clusters, n_genes)
	genes = np.unique(genes)  # 1d array of unique genes, sorted
	n_genes = len(genes)
	n_clusters = dsagg.shape[1]
	clusterborders = np.cumsum(dsagg.col_attrs["NCells"])

	# Now sort the genes by cluster enrichment
	top_cluster = []
	for g in genes:
		top_cluster.append(np.argsort(-dsagg["enrichment"][ds.ra.Gene == g, :][0])[0])
	genes = genes[np.argsort(top_cluster)]
	top_cluster = np.sort(top_cluster)

	plt.figure(figsize=(12, n_genes // 10))
	for ix, g in enumerate(genes):
		ax = plt.subplot(n_genes, 1, ix + 1)
		gix = np.where(ds.ra.Gene == g)[0][0]
		vals = ds[layer][gix, :]
		vals = vals / (np.percentile(vals, 99) + 0.1)
		ax.imshow(np.expand_dims(vals, axis=0), aspect='auto', cmap="viridis", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0, 0.9, g, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=4, color="black")
		text = plt.text(1.001, 0.9, g, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=4, color="black")

		cluster = top_cluster[ix]
		if cluster < len(clusterborders) - 1:
			xpos = clusterborders[cluster]
			text = plt.text(0.001 + xpos, -0.5, g, horizontalalignment='left', verticalalignment='top', fontsize=4, color="white")

		# Draw border between clusters
		if n_clusters > 2:
			tops = np.vstack((clusterborders - 0.5, np.zeros(clusterborders.shape[0]) - 0.5)).T
			bottoms = np.vstack((clusterborders - 0.5, np.zeros(clusterborders.shape[0]) + 0.5)).T
			lc = LineCollection(zip(tops, bottoms), linewidths=0.5, color='white', alpha=0.25)
			ax.add_collection(lc)

		if ix == 0:
			# Cluster IDs
			labels = ["#" + str(x) for x in np.arange(n_clusters)]
			if "ClusterName" in ds.ca:
				labels = dsagg.ca.ClusterName
			for ix in range(0, clusterborders.shape[0]):
				left = 0 if ix == 0 else clusterborders[ix - 1]
				right = clusterborders[ix]
				text = plt.text(left + (right - left) / 2, -1.5, labels[ix], horizontalalignment='center', verticalalignment='top', fontsize=4, color="black")

	plt.subplots_adjust(hspace=0)

	if out_file_root is not None:
		plt.savefig(out_file_root + "_TFs_heatmap.pdf", dpi=144)
	plt.close()

	n_cols = 10
	n_rows = math.ceil(len(genes) / 10)
	plt.figure(figsize=(15, 1.5 * n_rows))
	for i, gene in enumerate(genes):
		plt.subplot(n_rows, n_cols, i + 1)
		color = ds["pooled"][ds.ra.Gene == gene, :][0, :]
		cells = color > 0
		plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c="lightgrey", lw=0, marker='.', s=10, alpha=0.5)
		plt.scatter(ds.ca.TSNE[:, 0][cells], ds.ca.TSNE[:, 1][cells], c=color[cells], lw=0, marker='.', s=10, alpha=0.5)
		# Outline the cluster
		points = ds.ca.TSNE[ds.ca.Clusters == top_cluster[i], :]
		points = points[~mad(points), :]  # Remove outliers to get a tighter outline
		if points.shape[0] > 10:
			hull = ConvexHull(points)  # Find the convex hull
			plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], edgecolor="red", lw=1, fill=False)
		# Plot the gene name
		plt.text(0, ds.ca.TSNE[:, 1].min() * 1.05, gene, color="black", fontsize=10, horizontalalignment="center", verticalalignment="top")
		plt.axis("off")

	if out_file_root is not None:
		plt.savefig(out_file_root + "_TFs_scatter.png", dpi=144)
	plt.close()
