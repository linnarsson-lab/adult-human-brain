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


def plot_classes(ds: loompy.LoomConnection, out_file: str) -> None:
	n_cells = ds.shape[1]
	valid = ds.col_attrs["_Valid"].astype('bool')
	(a, b, w) = ds.get_edges("MKNN", axis=1)
	mknn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells)).tocsr()[valid, :][:, valid]
	sfdp = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[valid, :]
	labels = ds.col_attrs["Clusters"][valid]

	fig = plt.figure(figsize=(10, 20))
	g = nx.from_scipy_sparse_matrix(mknn)
	classes = ["Neurons", "Oligos", "Astrocyte", "Cycling", "Vascular", "Immune", "Ependymal"]
	colors = [plt.cm.get_cmap('Vega10')((ix + 0.5) / 10) for ix in range(10)]

	combined_colors = np.zeros((ds.shape[1], 4)) + np.array((0.5, 0.5, 0.5, 0))
	
	for ix, cls in enumerate(classes):
		ax = fig.add_subplot(4, 2, ix + 1)
		cmap = LinearSegmentedColormap.from_list('custom cmap', [(1, 1, 1, 0), colors[ix]])
		ax.set_title("P(" + classes[ix] + ")")
		nx.draw_networkx_edges(g, pos=sfdp, alpha=0.2, width=0.1, edge_color='gray')
		nx.draw_networkx_nodes(g, pos=sfdp, node_color=ds.col_attrs["Class_" + classes[ix]][valid], node_size=10, alpha=0.4, linewidths=0, cmap=cmap)
		ax.axis('off')
		cells = ds.col_attrs["Class"] == classes[ix]
		if np.sum(cells) > 0:
			combined_colors[cells] = [cmap(x) for x in ds.col_attrs["Class_" + classes[ix]][cells]]

	ax = fig.add_subplot(4, 2, 8)
	ax.set_title("Class")
	nx.draw_networkx_edges(g, pos=sfdp, alpha=0.2, width=0.1, edge_color='gray')
	nx.draw_networkx_nodes(g, pos=sfdp, node_color=combined_colors[valid], node_size=10, alpha=0.4, linewidths=0)
	ax.axis('off')

	plt.tight_layout()
	fig.savefig(out_file, format="png", dpi=300)
	plt.close()
