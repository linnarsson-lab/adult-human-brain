from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from sklearn.neighbors import NearestNeighbors

import loompy

from .colors import colors75


def manifold(ds: loompy.LoomConnection, out_file: str, tags: List[str] = None, embedding: str = "TSNE") -> None:
	n_cells = ds.shape[1]
	has_edges = False
	if "RNN" in ds.col_graphs:
		g = ds.col_graphs.RNN
		has_edges = True
	elif "MKNN" in ds.col_graphs:
		g = ds.col_graphs.MKNN
		has_edges = True
	if embedding in ds.ca:
		pos = ds.ca[embedding]
	else:
		raise ValueError("Embedding not found in the file")
	labels = ds.ca["Clusters"]
	if "Outliers" in ds.col_attrs:
		outliers = ds.col_attrs["Outliers"]
	else:
		outliers = np.zeros(ds.shape[1])
	# Compute a good size for the markers, based on local density
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
		lc = LineCollection(zip(pos[g.row], pos[g.col]), linewidths=0.25, zorder=0, color='thistle', alpha=0.1)
		ax.add_collection(lc)

	# Draw nodes
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
			plots.append(plt.scatter(x=pos[cluster, 0], y=pos[cluster, 1], c=[colors75[np.mod(i, 75)]], marker='.', lw=0, s=epsilon, alpha=0.5))
			txt = str(i)
			if "ClusterName" in ds.ca:
				txt = ds.ca.ClusterName[ds.ca["Clusters"] == i][0]
			if tags is not None:
				names.append(f"{txt}/n={n_cells} " + tags[i].replace("\n", " "))
			else:
				names.append(f"{txt}/n={n_cells}")
	plt.legend(plots, names, scatterpoints=1, markerscale=2, loc='upper left', bbox_to_anchor=(1, 1), fancybox=True, framealpha=0.5, fontsize=10)

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
	fig.savefig(out_file, format="png", dpi=144, bbox_inches='tight')
	plt.close()
