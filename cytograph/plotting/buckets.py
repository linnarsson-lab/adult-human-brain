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
