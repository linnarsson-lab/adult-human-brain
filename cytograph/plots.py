from typing import *
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import networkx as nx
import cytograph as cg
import luigi
import loompy
from palettable.tableau import Tableau_20
from matplotlib.colors import LinearSegmentedColormap
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects


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


def plot_graph(ds: loompy.LoomConnection, out_file: str, tags: List[str]) -> None:
	n_cells = ds.shape[1]
	valid = ds.col_attrs["_Valid"].astype('bool')
	(a, b, w) = ds.get_edges("MKNN", axis=1)
	mknn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells)).tocsr()[valid, :][:, valid]
	sfdp = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[valid, :]
	labels = ds.col_attrs["Clusters"][valid]

	fig = plt.figure(figsize=(10, 10))
	g = nx.from_scipy_sparse_matrix(mknn)
	ax = fig.add_subplot(111)

	# Draw the KNN graph first, with gray transparent edges
	nx.draw_networkx_edges(g, pos=sfdp, alpha=0.1, width=0.1, edge_color='gray')
	# Then draw the nodes, colored by label
	block_colors = (np.array(Tableau_20.colors) / 255)[np.mod(labels, 20)]
	nx.draw_networkx_nodes(g, pos=sfdp, node_color=block_colors, node_size=10, alpha=0.5, linewidths=0)

	mg_pos = []
	for lbl in range(0, max(labels) + 1):
		if np.sum(labels == lbl) == 0:
			continue
		(x, y) = np.median(sfdp[np.where(labels == lbl)[0]], axis=0)
		mg_pos.append((x, y))
		text = "#" + str(lbl)
		if len(tags[lbl]) > 0:
			text += "\n" + tags[lbl]
		ax.text(x, y, text, fontsize=6, bbox=dict(facecolor='gray', alpha=0.3, ec='none'))
	ax.axis('off')
	plt.tight_layout()
	fig.savefig(out_file, format="png", dpi=300)
	plt.close()


def plot_graph_age(ds: loompy.LoomConnection, out_file: str, tags: List[str]) -> None:
	def parse_age(age: str) -> float:
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


def plot_markerheatmap(ds: loompy.LoomConnection, out_file: str) -> None:
	n_markers = 10
	(markers, enrichment) = cg.MarkerSelection(n_markers=n_markers).fit(ds)

	# Load data and aggregate by cluster ID
	genes = markers
	cells = ds.col_attrs["Clusters"] >= 0
	data = np.log(ds[:, :][genes, :][:, cells] + 1)
	agg = npg.aggregate(ds.col_attrs["Clusters"][cells], data, axis=1)

	# Agglomerate cells
	zx = hc.ward(agg.T)
	xordering = hc.leaves_list(zx)

	# Reorder the cells according to the cluster ordering
	ordered = data[:, np.argsort(np.argsort(xordering)[ds.col_attrs["Clusters"][cells]])]
	# Reorder the genes according to the cluster ordering
	yordering = np.argsort(np.repeat(np.argsort(xordering), n_markers))
	ordered = ordered[yordering, :]

	classes = [x for x in ds.col_attrs.keys() if x.startswith("Class_")]
	n_classes = len(classes)

	topmarkers = ordered / np.max(ordered, axis=1)[None].T
	n_topmarkers = topmarkers.shape[0]

	fig = plt.figure(figsize=(30, 2.5 + n_classes / 5 + n_topmarkers / 10))
	gs = gridspec.GridSpec(2 + n_classes + 1, 1, height_ratios=[50, 1] + [1] * n_classes + [0.5 * n_topmarkers])

	ax = fig.add_subplot(gs[0])
	_ = hc.dendrogram(zx, no_labels=True, ax=ax)
	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	ax = fig.add_subplot(gs[1])
	ax.imshow(np.expand_dims(ds.col_attrs["_Total"], axis=0), aspect='auto', cmap="Reds")
	plt.text(0.001, 0.9, "Number of molecules", horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=9, color="black")
	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	for ix, cls in enumerate(classes):
		ax = fig.add_subplot(gs[2 + ix])
		ax.imshow(np.expand_dims(ds.col_attrs[cls][xordering], axis=0), aspect='auto', cmap="bone", vmin=0, vmax=1)
		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])
		text = plt.text(0.001, 0.9, cls[6:], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=10, color="white", weight="bold")
		text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
		
	ax = fig.add_subplot(gs[2 + n_classes])
	ax.imshow(topmarkers, aspect='auto', cmap="viridis", vmin=0, vmax=1)
	for ix in range(n_topmarkers):
		plt.text(0.001, 1 - (ix / n_topmarkers), ds.Gene[genes][yordering][ix], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=6, color="white")
	ax.set_frame_on(False)
	ax.set_xticks([])
	ax.set_yticks([])

	plt.subplots_adjust(hspace=0)
	plt.savefig(out_file, format="pdf", dpi=144)
	plt.close()