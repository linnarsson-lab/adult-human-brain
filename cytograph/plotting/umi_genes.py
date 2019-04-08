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
