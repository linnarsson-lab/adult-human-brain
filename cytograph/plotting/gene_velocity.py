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
from .utils import species
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.spatial import ConvexHull
from .midpoint_normalize import MidpointNormalize
from .colors import colorize


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
	plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(c), marker='.', s=10)
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
	plt.scatter(s, u, c=colorize(c), marker='.', s=10)
	maxs = np.max(s)
	plt.plot([0, maxs], [0, maxs * g], 'r--', color='b')
	plt.title("Phase portrait")		
	
	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()
