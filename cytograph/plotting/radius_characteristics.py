from typing import *
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import math
import networkx as nx
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
from cytograph.species import Species
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.spatial import ConvexHull
from .midpoint_normalize import MidpointNormalize


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
