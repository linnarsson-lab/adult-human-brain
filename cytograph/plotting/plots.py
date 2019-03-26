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


def manifold(ds: loompy.LoomConnection, out_file: str, tags: List[str] = None, embedding: str = "TSNE") -> None:
	logging.info("Loading graph")
	n_cells = ds.shape[1]
	cells = np.where(ds.ca["_Valid", "Valid"] == 1)[0]
	has_edges = False
	if "RNN" in ds.list_edges(axis=1):
		(a, b, w) = ds.get_edges("RNN", axis=1)
		has_edges = True
	elif "MKNN" in ds.list_edges(axis=1):
		(a, b, w) = ds.get_edges("MKNN", axis=1)
		has_edges = True
	if embedding == "TSNE":
		if "TSNE" in ds.ca:
			pos = ds.ca.TSNE
		else:
			pos = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()
	elif embedding in ds.ca:
		pos = ds.ca[embedding]
	else:
		raise ValueError("Embedding not found in the file")
	labels = ds.ca["Clusters"]
	if "Outliers" in ds.col_attrs:
		outliers = ds.col_attrs["Outliers"]
	else:
		outliers = np.zeros(ds.shape[1])
	# Compute a good size for the markers, based on local density
	logging.info("Computing node size")
	min_pts = 50
	eps_pct = 60
	nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
	nn.fit(pos)
	knn = nn.kneighbors_graph(mode='distance')
	k_radius = knn.max(axis=1).toarray()
	epsilon = (2500 / (pos.max() - pos.min())) * np.percentile(k_radius, eps_pct)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)

	# Draw edges
	if has_edges:
		logging.info("Drawing edges")
		lc = LineCollection(zip(pos[a], pos[b]), linewidths=0.25, zorder=0, color='thistle', alpha=0.1)
		ax.add_collection(lc)

	# Draw nodes
	logging.info("Drawing nodes")
	plots = []
	names = []
	for i in range(max(labels) + 1):
		cluster = labels == i
		n_cells = cluster.sum()
		if np.all(outliers[labels == i] == 1):
			edgecolor = colorConverter.to_rgba('red', alpha=.1)
			plots.append(plt.scatter(x=pos[outliers == 1, 0], y=pos[outliers == 1, 1], c='grey', marker='.', edgecolors=edgecolor, alpha=0.1, s=epsilon))
			names.append(f"{i}/n={n_cells}  (outliers)")
		else:
			plots.append(plt.scatter(x=pos[cluster, 0], y=pos[cluster, 1], c=[cg.colors75[np.mod(i, 75)]], marker='.', lw=0, s=epsilon, alpha=0.5))
			txt = str(i)
			if "ClusterName" in ds.ca:
				txt = ds.ca.ClusterName[ds.ca["Clusters"] == i][0]
			if tags is not None:
				names.append(f"{txt}/n={n_cells} " + tags[i].replace("\n", " "))
			else:
				names.append(f"{txt}/n={n_cells}")
	logging.info("Drawing legend")
	plt.legend(plots, names, scatterpoints=1, markerscale=2, loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, framealpha=0.5, fontsize=10)

	logging.info("Drawing cluster IDs")
	for lbl in range(0, max(labels) + 1):
		txt = str(lbl)
		if "ClusterName" in ds.ca:
			txt = ds.ca.ClusterName[ds.ca["Clusters"] == lbl][0]
		if np.all(outliers[labels == lbl] == 1):
			continue
		if np.sum(labels == lbl) == 0:
			continue
		(x, y) = np.median(pos[np.where(labels == lbl)[0]], axis=0)
		ax.text(x, y, txt, fontsize=12, bbox=dict(facecolor='white', alpha=0.5, ec='none'))
	plt.axis("off")
	logging.info("Saving to file")
	fig.savefig(out_file, format="png", dpi=144, bbox_inches='tight')
	plt.close()


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


def factors(ds: loompy.LoomConnection, base_name: str) -> None:
	logging.info(f"Plotting factors")
	offset = 0
	theta = ds.ca.HPF
	beta = ds.ra.HPF
	if "HPFVelocity" in ds.ca:
		v_hpf = ds.ca.HPFVelocity
	n_factors = theta.shape[1]
	while offset < n_factors:
		fig = plt.figure(figsize=(15, 15))
		fig.subplots_adjust(hspace=0, wspace=0)
		for nnc in range(offset, offset + 8):
			if nnc >= n_factors:
				break
			ax = plt.subplot(4, 4, (nnc - offset) * 2 + 1)
			plt.xticks(())
			plt.yticks(())
			plt.axis("off")
			plt.scatter(x=ds.ca.TSNE[:, 0], y=ds.ca.TSNE[:, 1], c='lightgrey', marker='.', alpha=0.5, s=60, lw=0)
			cells = theta[:, nnc] > np.percentile(theta[:, nnc], 99) * 0.1
			cmap = "viridis"
			plt.scatter(x=ds.ca.TSNE[cells, 0], y=ds.ca.TSNE[cells, 1], c=theta[:, nnc][cells], marker='.', alpha=0.5, s=60, cmap=cmap, lw=0)
			ax.text(.01, .99, '\n'.join(ds.ra.Gene[np.argsort(-beta[:, nnc])][:9]), horizontalalignment='left', verticalalignment="top", transform=ax.transAxes)
			ax.text(.99, .9, f"{nnc}", horizontalalignment='right', transform=ax.transAxes, fontsize=12)
			if "HPFVelocity" in ds.ca:
				ax.text(.5, .9, f"{np.percentile(v_hpf[nnc], 98):.2}", horizontalalignment='right', transform=ax.transAxes)
				vcmap = LinearSegmentedColormap.from_list("", ["red","whitesmoke","green"])
				norm = MidpointNormalize(midpoint=0)
				ax = plt.subplot(4, 4, (nnc - offset) * 2 + 2)
				plt.scatter(ds.ca.TSNE[:,0], ds.ca.TSNE[:,1],vmin=np.percentile(v_hpf[:,nnc], 2),vmax=np.percentile(v_hpf[:,nnc], 98), c=v_hpf[:,nnc],norm=norm, cmap=vcmap, marker='.',s=10)
				plt.axis("off")
		plt.savefig(base_name + f"{offset}.png", dpi=144)
		offset += 8
		plt.close()
		

def markers(ds: loompy.LoomConnection, out_file: str) -> None:
	xy = ds.ca.TSNE
	labels = ds.ca.Clusters
	if "pooled" in ds.layers:
		layer = ds["pooled"]
	else:
		layer = ds[""]
	markers: Dict[str, List[str]] = {}
	if species(ds) == "Homo sapiens":
		markers = {
			"Neuron": ["RBFOX1", "RBFOX3", "MEG3"],
			"Epen": ["CCDC153", "FOXJ1"],
			"Choroid": ["TTR"],
			"Astro": ["GJA1", "AQP4"],
			"Rgl": ["FABP7", "HOPX"],
			"Nblast": ["NHLH1"],
			"Endo": ["CLDN5"],
			"Immune": ["AIF1", "HEXB", "MRC1"],
			"OPC": ["PDGFRA", "CSPG4"],
			"Oligo": ["SOX10", "MOG"],
			"VLMC": ["LUM", "DCN", "COL1A1"],
			"Pericyte": ["VTN"],
			"VSM": ["ACTA2"]
		}
	elif species(ds) == "Mus musculus":
		markers = {
			"Neuron": ["Rbfox1", "Rbfox3", "Meg3"],
			"Epen": ["Ccdc153", "Foxj1"],
			"Choroid": ["Ttr"],
			"Astro": ["Gja1", "Aqp4"],
			"Rgl": ["Fabp7", "Hopx"],
			"Nblast": ["Nhlh1"],
			"Endo": ["Cldn5"],
			"Immune": ["Aif1", "Hexb", "Mrc1"],
			"OPC": ["Pdgfra", "Cspg4"],
			"Oligo": ["Sox10", "Mog"],
			"VLMC": ["Lum", "Dcn", "Col1a1"],
			"Pericyte": ["Vtn"],
			"VSM": ["Acta2"]
		}
	else:
		return
	plt.figure(figsize=(10, 10))
	ix = 1
	for celltype in markers.keys():
		for g in markers[celltype]:
			expr = layer[ds.ra.Gene == g, :][0]
			ax = plt.subplot(5, 5, ix)
			ax.axis("off")
			plt.title(celltype + ": " + g)
			plt.scatter(xy[:, 0], xy[:, 1], c="lightgrey", marker='.', lw=0, s=20)
			cells = expr > 0
			plt.scatter(xy[:, 0][cells], xy[:, 1][cells], c=expr[cells], marker='.', lw=0, s=20, cmap="viridis")
			ix += 1
	plt.savefig(out_file, format="png", dpi=144)
	plt.close()


def radius_characteristics(ds: loompy.LoomConnection, out_file: str = None) -> None:
	radius = 0.4
	if "radius" in ds.attrs:
		radius = ds.attrs.radius
	knn = ds.col_graphs.KNN
	knn.setdiag(0)
	dmin = 1 - knn.max(axis=1).toarray()[:, 0]  # Convert to distance since KNN uses similarities
	knn = sparse.coo_matrix((1 - knn.data, (knn.row, knn.col)), shape=knn.shape)
	knn.setdiag(0)
	dmax = knn.max(axis=1).toarray()[:, 0]
	knn = ds.col_graphs.KNN
	knn.setdiag(0)

	xy = ds.ca.TSNE

	cells = dmin < radius
	n_cells_inside = cells.sum()
	n_cells = dmax.shape[0]
	cells_pct = int(100 - 100 * (n_cells_inside / n_cells))
	n_edges_outside = (knn.data < 1 - radius).sum()
	n_edges = (knn.data > 0).sum()
	edges_pct = int(100 * (n_edges_outside / n_edges))

	plt.figure(figsize=(12, 12))
	plt.suptitle(f"Neighborhood characteristics (radius = {radius:.02})\n{n_cells - n_cells_inside} of {n_cells} cells lack neighbors ({cells_pct}%)\n{n_edges_outside} of {n_edges} edges removed ({edges_pct}%)", fontsize=14)

	ax = plt.subplot(321)
	ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey',s=1)
	cax = ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=dmax[cells], vmax=radius, cmap="viridis_r", s=1)
	plt.colorbar(cax)
	plt.title("Distance to farthest neighbor")

	ax = plt.subplot(322)
	ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', s=1)
	ax.scatter(xy[:, 0][~cells], xy[:, 1][~cells], c="red", s=1)
	plt.title("Cells with no neighbors inside radius")

	ax = plt.subplot(323)
	ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', s=1)
	subset = np.random.choice(np.sum(knn.data > 1 - radius), size=500)
	lc = LineCollection(zip(xy[knn.row[knn.data > 1 - radius]][subset], xy[knn.col[knn.data > 1 - radius]][subset]), linewidths=0.5, color="red")
	ax.add_collection(lc)
	plt.title("Edges inside radius (500 samples)")

	ax = plt.subplot(324)
	ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', s=1)
	subset = np.random.choice(np.sum(knn.data < 1 - radius), size=500)
	lc = LineCollection(zip(xy[knn.row[knn.data < 1 - radius]][subset], xy[knn.col[knn.data < 1 - radius]][subset]), linewidths=0.5, color="red")
	ax.add_collection(lc)
	plt.title("Edges outside radius (500 samples)")

	ax = plt.subplot(325)
	knn = ds.col_graphs.KNN
	d = 1 - knn.data
	d = d[d < 1]
	hist = plt.hist(d, bins=200)
	plt.ylabel("Number of cells")
	plt.xlabel("Jensen-Shannon distance to neighbors")
	plt.title(f"90th percentile JSD={radius:.2}")
	plt.plot([radius, radius], [0, hist[0].max()], "r--")

	plt.subplot(326)
	hist2 = plt.hist(dmax, bins=100, range=(0, 1), alpha=0.5)
	hist3 = plt.hist(dmin, bins=100, range=(0, 1), alpha=0.5)
	plt.title("Distance to nearest and farthest neighbors")
	plt.plot([radius, radius], [0, max(hist2[0].max(), hist3[0].max())], "r--")
	plt.ylabel("Number of cells")
	plt.xlabel("Jensen-Shannon distance to neighbors")

	if out_file is not None:
		plt.savefig(out_file, format="png", dpi=144)
	plt.close()


def batch_covariates(ds: loompy.LoomConnection, out_file: str) -> None:
	xy = ds.ca.TSNE
	plt.figure(figsize=(12, 12))

	if "Tissue" in ds.ca:
		labels = ds.ca.Tissue
	else:
		labels = np.array(["(unknown)"] * ds.shape[1])
	ax = plt.subplot(221)
	for lbl in np.unique(labels):
		cells = labels == lbl
		ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=cg.colorize(labels)[cells], label=lbl, lw=0, s=10)
	ax.legend()
	plt.title("Tissue")

	if "PCW" in ds.ca:
		labels = ds.ca.PCW
	elif "Age" in ds.ca:
		labels = ds.ca.Age
	else:
		labels = np.array(["(unknown)"] * ds.shape[1])
	ax = plt.subplot(222)
	for lbl in np.unique(labels):
		cells = labels == lbl
		ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=cg.colorize(labels)[cells], label=lbl, lw=0, s=0)
	cells = np.random.permutation(labels.shape[0])
	ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=cg.colorize(labels)[cells], lw=0, s=10)
	lgnd = ax.legend()
	for handle in lgnd.legendHandles:
		handle.set_sizes([10])
	plt.title("Age")
	
	if "XIST" in ds.ra.Gene:
		xist = ds[ds.ra.Gene == "XIST", :][0]
	elif "Xist" in ds.ra.Gene:
		xist = ds[ds.ra.Gene == "Xist", :][0]
	else:
		xist = np.array([0] * ds.shape[1])
	ax = plt.subplot(223)
	cells = xist > 0
	ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', lw=0, s=10)
	ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=xist[cells], lw=0, s=10)
	plt.title("Sex (XIST)")

	if "SampleID" in ds.ca:
		labels = ds.ca.SampleID
	else:
		labels = np.array(["(unknown)"] * ds.shape[1])
	ax = plt.subplot(224)
	for lbl in np.unique(labels):
		cells = labels == lbl
		ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=cg.colorize(labels)[cells], label=lbl, lw=0, s=10)
	ax.legend()
	plt.title("SampleID")

	plt.savefig(out_file, dpi=144)
	plt.close()

def umi_genes(ds: loompy.LoomConnection, out_file: str) -> None:
	plt.figure(figsize=(16, 4))
	plt.subplot(131)
	for chip in np.unique(ds.ca.SampleID):
		cells = ds.ca.SampleID == chip
		plt.hist(ds.ca.TotalRNA[cells], bins=100, label=chip, alpha=0.5, range=(0, 30000))
		plt.title("UMI distribution")
		plt.ylabel("Number of cells")
		plt.xlabel("Number of UMIs")
	plt.legend()
	plt.subplot(132)
	for chip in np.unique(ds.ca.SampleID):
		cells = ds.ca.SampleID == chip
		plt.hist(ds.ca.NGenes[cells], bins=100, label=chip, alpha=0.5, range=(0, 10000))
		plt.title("Gene count distribution")
		plt.ylabel("Number of cells")
		plt.xlabel("Number of genes")
	plt.legend()
	plt.subplot(133)
	tsne = ds.ca.TSNE
	plt.scatter(tsne[:, 0], tsne[:, 1], c="lightgrey", lw=0, marker='.')
	for chip in np.unique(ds.ca.SampleID):
		cells = (ds.ca.DoubletFlag == 1) & (ds.ca.SampleID == chip)
		plt.scatter(tsne[:, 0][cells], tsne[:, 1][cells], label=chip, lw=0, marker='.')
		plt.title("Doublets")
	plt.axis("off")
	plt.legend()
	plt.savefig(out_file, dpi=144)
	plt.close()


def gene_velocity(ds: loompy.LoomConnection, gene: str, out_file: str = None) -> None:
	genes = ds.ra.Gene[ds.ra.Selected == 1]
	g = ds.gamma[ds.ra.Gene == gene]
	s = ds["spliced_exp"][ds.ra.Gene == gene, :][0]
	u = ds["unspliced_exp"][ds.ra.Gene == gene, :][0]
	v = ds["velocity"][ds.ra.Gene == gene, :][0]
	c = ds.ca.Clusters
	vcmap = plt.colors.LinearSegmentedColormap.from_list("", ["red", "whitesmoke", "green"])

	plt.figure(figsize=(16, 4))
	plt.suptitle(gene)
	plt.subplot(141)
	plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=cg.colorize(c), marker='.', s=10)
	plt.title("Clusters")
	plt.axis("off")
	plt.subplot(142)
	norm = MidpointNormalize(midpoint=0)
	plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=v, norm=norm, cmap=vcmap, marker='.', s=10)
	plt.title("Velocity")
	plt.axis("off")
	plt.subplot(143)
	plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=s, cmap="viridis", marker='.',s=10)
	plt.title("Expression")
	plt.axis("off")
	plt.subplot(144)
	plt.scatter(s, u, c=cg.colorize(c), marker='.', s=10)
	maxs = np.max(s)
	plt.plot([0, maxs], [0, maxs * g], 'r--', color='b')
	plt.title("Phase portrait")		
	
	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()


def embedded_velocity(ds: loompy.LoomConnection, out_file: str = None) -> None:
	plt.figure(figsize=(12, 12))
	plt.subplot(221)
	xy = ds.ca.UMAP
	uv = ds.ca.UMAPVelocity
	plt.scatter(xy[:, 0], xy[:, 1], c=cg.colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=2.5, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (UMAP, cells)")

	plt.subplot(222)
	xy = ds.attrs.UMAPVelocityPoints
	uv = ds.attrs.UMAPVelocity
	plt.scatter(ds.ca.UMAP[:, 0], ds.ca.UMAP[:, 1], c=cg.colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=1, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (UMAP, grid)")

	plt.subplot(223)
	xy = ds.ca.TSNE
	uv = ds.ca.TSNEVelocity
	plt.scatter(xy[:, 0], xy[:, 1], c=cg.colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=2.5, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (TSNE, cells)")

	plt.subplot(224)
	xy = ds.attrs.TSNEVelocityPoints
	uv = ds.attrs.TSNEVelocity
	plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=cg.colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=1, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (TSNE, grid)")

	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()


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


def buckets(ds: loompy.LoomConnection, out_file: str = None) -> None:
	fig = plt.figure(figsize=(21, 7))
	plt.subplot(131)
	buckets = np.unique(ds.ca.Bucket)
	colors = cg.colorize(buckets)
	bucket_colors = {buckets[i]: colors[i] for i in range(len(buckets))}
	for ix, bucket in enumerate(np.unique(ds.ca.Bucket)):
		cells = ds.ca.Bucket == bucket
		plt.scatter(ds.ca.TSNE[cells, 0], ds.ca.TSNE[cells, 1], c=("lightgrey" if bucket == "" else colors[ix]), label=bucket, lw=0, marker='.', s=40, alpha=0.5)
		plt.axis("off")
	plt.legend(markerscale=3, loc="upper right")
	plt.title("Buckets from previous build")
	plt.subplot(132)
	cells = ds.ca.NewCells == 1
	plt.scatter(ds.ca.TSNE[~cells, 0], ds.ca.TSNE[~cells, 1], c="lightgrey", label="Old cells", lw=0, marker='.', s=40, alpha=0.5)
	plt.scatter(ds.ca.TSNE[cells, 0], ds.ca.TSNE[cells, 1], c="red", label="New cells", lw=0, marker='.', s=40, alpha=0.5)
	plt.axis("off")
	plt.legend(markerscale=3, loc="upper right")
	plt.title("Cells added in this build")
	plt.subplot(133)
	n_colors = len(bucket_colors)
	for ix, bucket in enumerate(np.unique(ds.ca.NewBucket)):
		cells = ds.ca.NewBucket == bucket
		color = "lightgrey"
		if bucket != "":
			if bucket in bucket_colors.keys():
				color = bucket_colors[bucket]
			else:
				color = cg.colors75[n_colors]
				bucket_colors[bucket] = color
				n_colors += 1
		plt.scatter(ds.ca.TSNE[cells, 0], ds.ca.TSNE[cells, 1], c=color, label=bucket, lw=0, marker='.', s=40, alpha=0.5)
		plt.axis("off")
	plt.legend(markerscale=3, loc="upper right")
	plt.title("Buckets proposed for this build")

	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()
