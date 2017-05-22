from typing import *
import os
import logging
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import networkx as nx
import cytograph as cg
import luigi
import loompy
from matplotlib.colors import LinearSegmentedColormap
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph


def plot_cv_mean(ds: loompy.LoomConnection, out_file: str) -> None:
	mu = ds.row_attrs["_LogMean"]
	cv = ds.row_attrs["_LogCV"]
	selected = ds.row_attrs["_Selected"].astype('bool')
	excluded = (1 - ds.row_attrs["_Valid"]).astype('bool')

	fig = plt.figure(figsize=(8, 6))
	ax1 = fig.add_subplot(111)
	h1 = ax1.scatter(mu, cv, c='grey', alpha=0.5, marker=".", edgecolors="none")
	h2 = ax1.scatter(mu[excluded], cv[excluded], alpha=0.5, marker=".", edgecolors="none")
	h3 = ax1.scatter(mu[selected], cv[selected], alpha=0.5, marker=".", edgecolors="none")
	plt.ylabel('Log(CV)')
	plt.xlabel('Log(Mean)')
	plt.legend([h1, h2, h3], ["Not selected", "Excluded", "Selected"])
	plt.tight_layout()
	fig.savefig(out_file, format="png", dpi=144)
	plt.close()


def plot_knn(ds: loompy.LoomConnection, out_file: str, tags: List[str]) -> None:
	n_cells = ds.shape[1]
	valid = ds.col_attrs["_Valid"].astype('bool')
	(a, b, w) = ds.get_edges("MKNN", axis=1)
	mknn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells)).tocsr()[valid, :][:, valid]
	xy = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[valid, :]

	fig = plt.figure(figsize=(10, 10))
	g = nx.from_scipy_sparse_matrix(mknn)
	ax = fig.add_subplot(111)

	nx.draw_networkx_edges(g, pos=xy, alpha=0.1, width=0.1, edge_color='gray')
	ax.axis('off')
	plt.tight_layout()
	fig.savefig(out_file, format="png", dpi=300)
	plt.close()


def plot_graph(ds: loompy.LoomConnection, out_file: str, tags: List[str] = None) -> None:
	logging.info("Loading graph")
	n_cells = ds.shape[1]
	cells = np.where(ds.col_attrs["_Valid"] == 1)[0]
	(a, b, w) = ds.get_edges("MKNN", axis=1)
	pos = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()
	labels = ds.col_attrs["Clusters"]
	outliers = ds.col_attrs["Outliers"]

	# Compute a good size for the markers, based on local density
	logging.info("Computing node size")
	min_pts = 50
	eps_pct = 60
	nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
	nn.fit(pos)
	knn = nn.kneighbors_graph(mode='distance')
	k_radius = knn.max(axis=1).toarray()
	epsilon = 24 * np.percentile(k_radius, eps_pct)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)

	# Draw edges
	logging.info("Drawing edges")
	lc = LineCollection(zip(pos[a], pos[b]), linewidths=0.5, zorder=0, color='black', alpha=0.25)
	ax.add_collection(lc)

	# Draw nodes
	logging.info("Drawing nodes")
	colors20 = np.vstack((plt.cm.Vega20b(np.linspace(0., 1, 20))[::2], plt.cm.Vega20c(np.linspace(0, 1, 20))[1::2]))
	plots = []
	names = []
	for i in range(max(labels) + 1):
		cluster = labels == i
		if np.all(outliers[labels == i] == 1):
			plots.append(plt.scatter(x=pos[outliers == 1, 0], y=pos[outliers == 1, 1], c='grey', marker='.', edgecolors='r', alpha=0.5, s=epsilon))
			names.append(str(i) + " (outliers)")
		else:
			plots.append(plt.scatter(x=pos[cluster, 0], y=pos[cluster, 1], c=colors20[np.mod(i, 20)], marker='.', lw=0, s=epsilon, alpha=0.75))
			if tags is not None:
				names.append(str(i) + " " + tags[i].replace("\n", " "))
			else:
				names.append(str(i))
	logging.info("Drawing legend")
	plt.legend(plots, names, scatterpoints=1, markerscale=2, loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, framealpha=0.5, fontsize=10)

	logging.info("Drawing cluster IDs")
	mg_pos = []
	for lbl in range(0, max(labels) + 1):
		if np.all(outliers[labels == lbl] == 1):
			continue
		if np.sum(labels == lbl) == 0:
			continue
		(x, y) = np.median(pos[np.where(labels == lbl)[0]], axis=0)
		mg_pos.append((x, y))
		ax.text(x, y, str(lbl), fontsize=12, bbox=dict(facecolor='white', alpha=0.5, ec='none'))
	logging.info("Saving to file")
	fig.savefig(out_file, format="png", dpi=144, bbox_inches='tight')
	plt.close()


def plot_graph_age(ds: loompy.LoomConnection, out_file: str, tags: List[str]) -> None:
	def parse_age(age: str) -> float:
		if age == "":
			return 0
		unit, amount = age[0], float(age[1:])
		if unit == "P":
			amount += 19.
		return amount
	
	n_cells = ds.shape[1]
	valid = ds.col_attrs["_Valid"].astype('bool')
	
	(a, b, w) = ds.get_edges("MKNN", axis=1)
	mknn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells)).tocsr()[valid, :][:, valid]
	sfdp = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[valid, :]
	# The sorting below is to make every circle visible and avoid overlappings in crowded situations
	orderx = np.argsort(sfdp[:, 0], kind="mergesort")
	ordery = np.argsort(sfdp[:, 1], kind="mergesort")
	orderfin = orderx[ordery]
	sfdp_original = sfdp.copy()  # still the draw_networkx_edges wants the sfd with respect of the graph `g`
	# \it is shortcut to avoid resorting the graph
	sfdp = sfdp[orderfin, :]
	labels = ds.col_attrs["Clusters"][valid][orderfin]
	age = np.fromiter(map(parse_age, ds.col_attrs["Age"]), dtype=float)[valid][orderfin]

	fig = plt.figure(figsize=(10, 10))
	g = nx.from_scipy_sparse_matrix(mknn)
	ax = fig.add_subplot(111)

	# Draw the KNN graph first, with gray transparent edges
	nx.draw_networkx_edges(g, pos=sfdp_original, alpha=0.1, width=0.1, edge_color='gray')
	# Then draw the nodes, colored by label
	block_colors = plt.cm.nipy_spectral_r((age - 6) / 14.)
	nx.draw_networkx_nodes(g, pos=sfdp, node_color=block_colors, node_size=10, alpha=0.4, linewidths=0)

	mg_pos = []
	for lbl in range(0, max(labels) + 1):
		if np.sum(labels == lbl) == 0:
			continue
		(x, y) = np.median(sfdp[np.where(labels == lbl)[0]], axis=0)
		mg_pos.append((x, y))
		text = "#" + str(lbl)
		if len(tags[lbl]) > 0:
			text += "\n" + tags[lbl]
		ax.text(x, y, text, fontsize=8, bbox=dict(facecolor='gray', alpha=0.3, ec='none'))
	ax.axis('off')
	levels = np.unique(age)
	for il, lev in enumerate(levels):
		ax.add_patch(
			plt.Rectangle(
				(0.90, 0.7 + il * 0.016), 0.014, 0.014,
				color=plt.cm.nipy_spectral_r((lev - 6) / 14.),
				clip_on=0, transform=ax.transAxes) )
		ax.text(0.93, 0.703 + il * 0.016, ("E%.1f" % lev if lev < 18.5 else "P%.1f" % (lev - 19)), transform=ax.transAxes)
	plt.tight_layout()
	fig.savefig(out_file, format="png", dpi=300)
	plt.close()


def plot_classification(ds: loompy.LoomConnection, out_file: str) -> None:
	n_cells = ds.shape[1]
	valid = ds.col_attrs["_Valid"].astype('bool')
	(a, b, w) = ds.get_edges("MKNN", axis=1)
	mknn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells)).tocsr()[valid, :][:, valid]
	pos = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[valid, :]
	labels = ds.col_attrs["Clusters"][valid]

	fig = plt.figure(figsize=(10, 10))
	g = nx.from_scipy_sparse_matrix(mknn)
	classes = ["Neurons", "Astrocyte", "Ependymal", "OEC", "Oligos", "Schwann", "Cycling", "Vascular", "Immune"]
	colors = [plt.cm.get_cmap('Vega20')((ix + 0.5) / 20) for ix in range(20)]

	combined_colors = np.zeros((ds.shape[1], 4)) + np.array((0.5, 0.5, 0.5, 0))
	
	for ix, cls in enumerate(classes):
		cmap = LinearSegmentedColormap.from_list('custom cmap', [(1, 1, 1, 0), colors[ix]])
		cells = ds.col_attrs["Class0"] == classes[ix]
		if np.sum(cells) > 0:
			combined_colors[cells] = [cmap(x) for x in ds.col_attrs["Class_" + classes[ix]][cells]]

	cmap = LinearSegmentedColormap.from_list('custom cmap', [(1, 1, 1, 0), colors[ix + 1]])
	ery_color = np.array([[1, 1, 1, 0], [0.9, 0.71, 0.76, 0]])[(ds.col_attrs["Class"][valid] == "Erythrocyte").astype('int')]
	cells = ds.col_attrs["Class0"] == "Erythrocyte"
	if np.sum(cells) > 0:
		combined_colors[cells] = np.array([1, 0.71, 0.76, 0])

	cmap = LinearSegmentedColormap.from_list('custom cmap', [(1, 1, 1, 0), colors[ix + 2]])
	exc_color = np.array([[1, 1, 1, 0], [0.5, 0.5, 0.5, 0]])[(ds.col_attrs["Class0"][valid] == "Excluded").astype('int')]
	cells = ds.col_attrs["Class0"] == "Excluded"
	if np.sum(cells) > 0:
		combined_colors[cells] = np.array([0.5, 0.5, 0.5, 0])

	ax = fig.add_subplot(1, 1, 1)
	ax.set_title("Class")
	nx.draw_networkx_edges(g, pos=pos, alpha=0.2, width=0.1, edge_color='gray')
	nx.draw_networkx_nodes(g, pos=pos, node_color=combined_colors[valid], node_size=10, alpha=0.6, linewidths=0)
	ax.axis('off')

	plt.tight_layout()
	fig.savefig(out_file, format="png", dpi=300)
	plt.close()


def plot_classes(ds: loompy.LoomConnection, out_file: str) -> None:
	n_cells = ds.shape[1]
	valid = ds.col_attrs["_Valid"].astype('bool')
	(a, b, w) = ds.get_edges("MKNN", axis=1)
	mknn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells)).tocsr()[valid, :][:, valid]
	sfdp = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[valid, :]
	labels = ds.col_attrs["Clusters"][valid]

	fig = plt.figure(figsize=(24, 18))
	g = nx.from_scipy_sparse_matrix(mknn)
	classes = ["Neurons", "Astrocyte", "Ependymal", "OEC", "Oligos", "Schwann", "Cycling", "Vascular", "Immune"]
	colors = [plt.cm.get_cmap('Vega20')((ix + 0.5) / 20) for ix in range(20)]

	combined_colors = np.zeros((ds.shape[1], 4)) + np.array((0.5, 0.5, 0.5, 0))
	
	for ix, cls in enumerate(classes):
		ax = fig.add_subplot(3, 4, ix + 1)
		cmap = LinearSegmentedColormap.from_list('custom cmap', [(1, 1, 1, 0), colors[ix]])
		ax.set_title("P(" + classes[ix] + ")")
		nx.draw_networkx_edges(g, pos=sfdp, alpha=0.2, width=0.1, edge_color='gray')
		nx.draw_networkx_nodes(g, pos=sfdp, node_color=ds.col_attrs["Class_" + classes[ix]][valid], node_size=10, alpha=0.6, linewidths=0, cmap=cmap)
		ax.axis('off')
		cells = ds.col_attrs["Class0"] == classes[ix]
		if np.sum(cells) > 0:
			combined_colors[cells] = [cmap(x) for x in ds.col_attrs["Class_" + classes[ix]][cells]]

	ax = fig.add_subplot(3, 4, ix + 2)
	cmap = LinearSegmentedColormap.from_list('custom cmap', [(1, 1, 1, 0), colors[ix + 1]])
	ax.set_title("Erythrocytes")
	nx.draw_networkx_edges(g, pos=sfdp, alpha=0.2, width=0.1, edge_color='gray')
	ery_color = np.array([[1, 1, 1, 0], [0.9, 0.71, 0.76, 0]])[(ds.col_attrs["Class"][valid] == "Erythrocyte").astype('int')]
	nx.draw_networkx_nodes(g, pos=sfdp, node_color=ery_color, node_size=10, alpha=0.6, linewidths=0, cmap=cmap)
	ax.axis('off')
	cells = ds.col_attrs["Class0"] == "Erythrocyte"
	if np.sum(cells) > 0:
		combined_colors[cells] = np.array([1, 0.71, 0.76, 0])

	ax = fig.add_subplot(3, 4, ix + 3)
	cmap = LinearSegmentedColormap.from_list('custom cmap', [(1, 1, 1, 0), colors[ix + 2]])
	ax.set_title("Excluded")
	nx.draw_networkx_edges(g, pos=sfdp, alpha=0.2, width=0.1, edge_color='gray')
	exc_color = np.array([[1, 1, 1, 0], [0.5, 0.5, 0.5, 0]])[(ds.col_attrs["Class0"][valid] == "Excluded").astype('int')]
	nx.draw_networkx_nodes(g, pos=sfdp, node_color=exc_color, node_size=10, alpha=0.6, linewidths=0, cmap=cmap)
	ax.axis('off')
	cells = ds.col_attrs["Class0"] == "Excluded"
	if np.sum(cells) > 0:
		combined_colors[cells] = np.array([0.5, 0.5, 0.5, 0])

	ax = fig.add_subplot(3, 4, 12)
	ax.set_title("Class")
	nx.draw_networkx_edges(g, pos=sfdp, alpha=0.2, width=0.1, edge_color='gray')
	nx.draw_networkx_nodes(g, pos=sfdp, node_color=combined_colors[valid], node_size=10, alpha=0.6, linewidths=0)
	ax.axis('off')

	plt.tight_layout()
	fig.savefig(out_file, format="png", dpi=300)
	plt.close()


def plot_markerheatmap(ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, n_markers_per_cluster: int = 10, out_file: str = None) -> None:
	n_clusters = np.max(dsagg.col_attrs["Clusters"] + 1)
	n_markers = n_markers_per_cluster * n_clusters
	enrichment = dsagg.layer["enrichment"][:n_markers, :]
	cells = ds.col_attrs["Clusters"] >= 0
	data = np.log(ds[:n_markers, :][:, cells] + 1)
	agg = np.log(dsagg[:n_markers, :] + 1)

	clusterborders = np.cumsum(dsagg.col_attrs["NCells"])
	gene_pos = clusterborders[np.argmax(enrichment, axis=1)]
	tissues = set(ds.col_attrs["Tissue"])
	n_tissues = len(tissues)

	classes = [x for x in ds.col_attrs.keys() if x.startswith("Class_")]
	n_classes = len(classes)

	genes = ["Cdk1", "Top2a", "Hexb", "Mrc1", "Lum", "Col1a1", "Cldn5", "Acta2", "Tagln", "Foxj1", "Aqp4", "Meg3", "Stmn2", "Gad2", "Slc32a1", "Plp1", "Sox10", "Mog", "Mbp", "Mpz"]
	n_genes = len(genes)

	colormax = np.percentile(data, 99, axis=1) + 0.1
	topmarkers = data / colormax[None].T
	n_topmarkers = topmarkers.shape[0]

	fig = plt.figure(figsize=(30, 4.5 + n_tissues / 5 + n_classes / 5 + n_genes / 5 + n_topmarkers / 10))
	gs = gridspec.GridSpec(3 + n_tissues + n_classes + n_genes + 1, 1, height_ratios=[1, 1, 1] + [1] * n_tissues + [1] * n_classes + [1] * n_genes + [0.5 * n_topmarkers])

	ax = fig.add_subplot(gs[1])
	ax.imshow(np.expand_dims(ds.col_attrs["Outliers"][cells], axis=0), aspect='auto', cmap="Reds")
	plt.text(0.001, 0.9, "Outliers", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=9, color="black")
	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	ax = fig.add_subplot(gs[2])
	ax.imshow(np.expand_dims(ds.col_attrs["_Total"][cells], axis=0), aspect='auto', cmap="Reds")
	plt.text(0.001, 0.9, "Number of molecules", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=9, color="black")
	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	for ix, t in enumerate(tissues):
		ax = fig.add_subplot(gs[3 + ix])
		ax.imshow(np.expand_dims((ds.col_attrs["Tissue"][cells] == t).astype('int'), axis=0), aspect='auto', cmap="bone", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0.001, 0.9, t, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white", weight="bold")
		text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

	for ix, cls in enumerate(classes):
		ax = fig.add_subplot(gs[3 + n_tissues + ix])
		ax.imshow(np.expand_dims(ds.col_attrs[cls][cells], axis=0), aspect='auto', cmap="bone", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0.001, 0.9, cls[6:], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white", weight="bold")
		text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])

	for ix, g in enumerate(genes):
		ax = fig.add_subplot(gs[3 + n_tissues + n_classes + ix])
		gix = np.where(ds.Gene == g)[0]
		vals = ds[gix, :][cells]
		vals = vals / (np.percentile(vals, 99) + 0.1)
		ax.imshow(np.expand_dims(vals, axis=0), aspect='auto', cmap="viridis", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0.001, 0.9, g, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white", weight="bold")

	ax = fig.add_subplot(gs[3 + n_tissues + n_classes + n_genes])
	# Draw border between clusters
	tops = np.vstack((clusterborders - 0.5, np.zeros(clusterborders.shape[0]) - 0.5)).T
	bottoms = np.vstack((clusterborders - 0.5, np.zeros(clusterborders.shape[0]) + n_topmarkers - 0.5)).T
	lc = LineCollection(zip(tops, bottoms), linewidths=1, color='white', alpha=0.5)
	ax.add_collection(lc)
		
	ax.imshow(topmarkers, aspect='auto', cmap="viridis", vmin=0, vmax=1)
	for ix in range(n_topmarkers):
		xpos = gene_pos[ix]
		if xpos == clusterborders[-1]:
			xpos = clusterborders[-3]
		text = plt.text(0.001 + xpos, ix - 0.5, ds.Gene[:n_markers][ix], horizontalalignment='left', verticalalignment='top', fontsize=4, color="white")

	# Cluster IDs
	for ix in range(1, clusterborders.shape[0]):
		text = plt.text(clusterborders[ix - 1] + (clusterborders[ix] - clusterborders[ix - 1]) / 2, 1, "#" + str(ix), horizontalalignment='center', verticalalignment='top', fontsize=6, color="white", weight="bold")

	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	plt.subplots_adjust(hspace=0)
	if out_file is not None:
		plt.savefig(out_file, format="pdf", dpi=144)
		plt.close()
