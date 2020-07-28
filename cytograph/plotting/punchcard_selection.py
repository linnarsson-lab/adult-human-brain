from typing import List
import matplotlib.pyplot as plt
import numpy as np
import loompy
from sklearn.neighbors import NearestNeighbors
from .colors import colorize, colors75
from scipy.stats import mode


def punchcard_selection(ds: loompy.LoomConnection, out_file: str = None, tag1: List[str] = None, tag2: List[str] = None) -> None:
	fig = plt.figure(None, (25, 10))
	ax = fig.add_axes([0, 0, 0.4, 1])
	pos = ds.ca.TSNE

	# Compute a good size for the markers, based on local density
	n_cells = ds.shape[1]
	min_pts = min(int(n_cells / 3), 50)
	eps_pct = 60
	nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
	nn.fit(pos)
	knn = nn.kneighbors_graph(mode='distance')
	k_radius = knn.max(axis=1).toarray()
	epsilon = (2500 / (pos.max() - pos.min())) * np.percentile(k_radius, eps_pct)

	# Get colors
	subsets = np.unique(ds.ca.Subset)
	colors = colorize(subsets)
	subset_colors = {subsets[i]: colors[i] for i in range(len(subsets))}

	# Draw nodes
	# Color cluster labels by most common subset in the cluster
	labels = ds.ca.Clusters
	plots = []
	tag1_names = []
	tag2_names = []
	for i in range(max(labels) + 1):
		cluster = labels == i
		n_cells = cluster.sum()
		subset = mode(ds.ca.Subset[labels == i])[0][0]
		c = [("lightgrey" if subset == "" else subset_colors[subset])] * n_cells
		plots.append(ax.scatter(x=pos[cluster, 0], y=pos[cluster, 1], c=c, marker='.', lw=0, s=epsilon, alpha=0.5))
		txt = str(i)
		if "ClusterName" in ds.ca:
			txt = ds.ca.ClusterName[ds.ca["Clusters"] == i][0]
		if tag1 is not None:
			tag1_names.append(f"{txt}/n={n_cells} " + tag1[i].replace("\n", " "))
		else:
			tag1_names.append(f"{txt}/n={n_cells}")
		if tag2 is not None:
			tag2_names.append(f"{txt} " + tag2[i].replace("\n", " "))

	# Add legends
	if ds.ca.Clusters.max() <= 500:
		ax2 = fig.add_axes([0.4, 0, 0.3, 1])
		ax2.axis("off")
		ax2.legend(plots, tag1_names, scatterpoints=1, markerscale=2, loc='center', mode='expand', fancybox=True, framealpha=0.5, fontsize=12)
		if tag2 is not None:
			ax3 = fig.add_axes([0.7, 0, 0.3, 1])
			ax3.axis("off")
			ax3.legend(plots, tag2_names, scatterpoints=1, markerscale=2, loc='center', mode='expand', fancybox=True, framealpha=0.5, fontsize=12)

	# Add key for subsets
	y = 0.99
	for subset in subsets:
		ax.text(.01, y, subset, color=subset_colors[subset], fontsize=12, horizontalalignment='left', verticalalignment="top", transform=ax.transAxes)
		y -= 0.03

	# Labeling clusters
	for lbl in range(0, max(labels) + 1):
		txt = str(lbl)
		(x, y) = np.median(pos[np.where(labels == lbl)[0]], axis=0)
		subset = mode(ds.ca.Subset[labels == lbl])[0][0]
		ax.text(x, y, txt, fontsize=12, bbox=dict(facecolor=subset_colors[subset], alpha=0.5, ec='none'))
	
	# Draw nodes again
	# Cells on the tSNE will be colored individually and not cluster-wise
	labels = ds.ca.Subset
	tag1_names = []
	tag2_names = []
	for i in np.unique(labels):
		cells = labels == i
		n_cells = cells.sum()
		subset = ds.ca.Subset[labels == i][0]
		c = [("lightgrey" if subset == "" else subset_colors[subset])] * n_cells
		ax.scatter(x=pos[cells, 0], y=pos[cells, 1], c=c, marker='.', lw=0, s=epsilon, alpha=0.5)
	ax.axis('off')

	if out_file is not None:
		fig.savefig(out_file, format="png", dpi=144, bbox_inches='tight')

	plt.close()
