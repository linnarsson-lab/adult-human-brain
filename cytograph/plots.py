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
		ax.text(x, y, text, fontsize=8, bbox=dict(facecolor='gray', alpha=0.3, ec='none'))
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
