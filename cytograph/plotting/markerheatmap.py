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


def markerheatmap(ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, n_markers_per_cluster: int = 10, out_file: str = None) -> None:
	logging.info(ds.shape)
	layer = "pooled" if "pooled" in ds.layers else ""
	n_clusters = np.max(dsagg.ca["Clusters"] + 1)
	n_markers = n_markers_per_cluster * n_clusters
	enrichment = dsagg.layer["enrichment"][:n_markers, :]
	cells = ds.ca["Clusters"] >= 0
	data = np.log(ds[layer][:n_markers, :][:, cells] + 1)
	agg = np.log(dsagg[:n_markers, :] + 1)

	clusterborders = np.cumsum(dsagg.col_attrs["NCells"])
	gene_pos = clusterborders[np.argmax(enrichment, axis=1)]
	tissues: Set[str] = set()
	if "Tissue" in ds.ca:
		tissues = set(ds.col_attrs["Tissue"])
	n_tissues = len(tissues)

	classes = []
	if "Subclass" in ds.ca:
		classes = sorted(list(set(ds.col_attrs["Subclass"])))
	n_classes = len(classes)

	probclasses = [x for x in ds.col_attrs.keys() if x.startswith("ClassProbability_")]
	n_probclasses = len(probclasses)

	gene_names: List[str] = []
	if species(ds) == "Mus musculus":
		gene_names = ["Pcna", "Cdk1", "Top2a", "Fabp7", "Fabp5", "Hopx", "Aif1", "Hexb", "Mrc1", "Lum", "Col1a1", "Cldn5", "Acta2", "Tagln", "Tmem212", "Foxj1", "Aqp4", "Gja1", "Rbfox1", "Eomes", "Gad1", "Gad2", "Slc32a1", "Slc17a7", "Slc17a8", "Slc17a6", "Tph2", "Fev", "Th", "Slc6a3", "Chat", "Slc5a7", "Slc18a3", "Slc6a5", "Slc6a9", "Dbh", "Slc18a2", "Plp1", "Sox10", "Mog", "Mbp", "Mpz", "Emx1", "Dlx5"]
	elif species(ds) == "Homo sapiens":
		gene_names = ["PCNA", "CDK1", "TOP2A", "FABP7", "FABP5", "HOPX", "AIF1", "HEXB", "MRC1", "LUM", "COL1A1", "CLDN5", "ACTA2", "TAGLN", "TMEM212", "FOXJ1", "AQP4", "GJA1", "RBFOX1", "EOMES", "GAD1", "GAD2", "SLC32A1", "SLC17A7", "SLC17A8", "SLC17A6", "TPH2", "FEV", "TH", "SLC6A3", "CHAT", "SLC5A7", "SLC18A3", "SLC6A5", "SLC6A9", "DBH", "SLC18A2", "PLP1", "SOX10", "MOG", "MBP", "MPZ", "EMX1", "DLX5"]
	genes = [g for g in gene_names if g in ds.ra.Gene]
	n_genes = len(genes)

	colormax = np.percentile(data, 99, axis=1) + 0.1
	# colormax = np.max(data, axis=1)
	topmarkers = data / colormax[None].T
	n_topmarkers = topmarkers.shape[0]

	fig = plt.figure(figsize=(30, 4.5 + n_tissues / 5 + n_classes / 5 + n_probclasses / 5 + n_genes / 5 + n_topmarkers / 10))
	gs = gridspec.GridSpec(3 + n_tissues + n_classes + n_probclasses + n_genes + 1, 1, height_ratios=[1, 1, 1] + [1] * n_tissues + [1] * n_classes + [1] * n_probclasses + [1] * n_genes + [0.5 * n_topmarkers])

	ax = fig.add_subplot(gs[1])
	if "Outliers" in ds.col_attrs:
		ax.imshow(np.expand_dims(ds.col_attrs["Outliers"][cells], axis=0), aspect='auto', cmap="Reds")
	plt.text(0.001, 0.9, "Outliers", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=9, color="black")
	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	ax = fig.add_subplot(gs[2])
	if "_Total" in ds.ca or "Total" in ds.ca:
		ax.imshow(np.expand_dims(ds.ca["_Total", "Total"][cells], axis=0), aspect='auto', cmap="Reds")
	plt.text(0.001, 0.9, "Number of molecules", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=9, color="black")
	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	for ix, t in enumerate(tissues):
		ax = fig.add_subplot(gs[3 + ix])
		ax.imshow(np.expand_dims((ds.ca["Tissue"][cells] == t).astype('int'), axis=0), aspect='auto', cmap="bone", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0.001, 0.9, t, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white", weight="bold")
		text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

	for ix, cls in enumerate(classes):
		ax = fig.add_subplot(gs[3 + n_tissues + ix])
		ax.imshow(np.expand_dims((ds.ca["Subclass"] == cls).astype('int')[cells], axis=0), aspect='auto', cmap="binary_r", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0.001, 0.9, cls, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white", weight="bold")
		text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

	for ix, cls in enumerate(probclasses):
		ax = fig.add_subplot(gs[3 + n_tissues + n_classes + ix])
		ax.imshow(np.expand_dims(ds.col_attrs[cls][cells], axis=0), aspect='auto', cmap="pink", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0.001, 0.9, "P(" + cls[17:] + ")", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white", weight="bold")
		text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

	for ix, g in enumerate(genes):
		ax = fig.add_subplot(gs[3 + n_tissues + n_classes + n_probclasses + ix])
		gix = np.where(ds.ra.Gene == g)[0][0]
		vals = ds[layer][gix, :][cells]
		vals = vals / (np.percentile(vals, 99) + 0.1)
		ax.imshow(np.expand_dims(vals, axis=0), aspect='auto', cmap="viridis", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0.001, 0.9, g, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white", weight="bold")

	ax = fig.add_subplot(gs[3 + n_tissues + n_classes + n_probclasses + n_genes])
	# Draw border between clusters
	if n_clusters > 2:
		tops = np.vstack((clusterborders - 0.5, np.zeros(clusterborders.shape[0]) - 0.5)).T
		bottoms = np.vstack((clusterborders - 0.5, np.zeros(clusterborders.shape[0]) + n_topmarkers - 0.5)).T
		lc = LineCollection(zip(tops, bottoms), linewidths=1, color='white', alpha=0.5)
		ax.add_collection(lc)

	ax.imshow(topmarkers, aspect='auto', cmap="viridis", vmin=0, vmax=1)
	for ix in range(n_topmarkers):
		xpos = gene_pos[ix]
		if xpos == clusterborders[-1]:
			if n_clusters > 2:
				xpos = clusterborders[-3]
		text = plt.text(0.001 + xpos, ix - 0.5, ds.ra.Gene[:n_markers][ix], horizontalalignment='left', verticalalignment='top', fontsize=4, color="white")

	# Cluster IDs
	labels = ["#" + str(x) for x in np.arange(n_clusters)]
	if "ClusterName" in ds.ca:
		labels = dsagg.ca.ClusterName
	for ix in range(0, clusterborders.shape[0]):
		left = 0 if ix == 0 else clusterborders[ix - 1]
		right = clusterborders[ix]
		text = plt.text(left + (right - left) / 2, 1, labels[ix], horizontalalignment='center', verticalalignment='top', fontsize=6, color="white", weight="bold")

	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	plt.subplots_adjust(hspace=0)
	if out_file is not None:
		plt.savefig(out_file, format="pdf", dpi=144)
	plt.close()
