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
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.spatial import ConvexHull
from .midpoint_normalize import MidpointNormalize
from .colors import colorize


def embedded_velocity(ds: loompy.LoomConnection, out_file: str = None) -> None:
	plt.figure(figsize=(12, 12))
	plt.subplot(221)
	xy = ds.ca.UMAP
	uv = ds.ca.UMAPVelocity
	plt.scatter(xy[:, 0], xy[:, 1], c=colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=2.5, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (UMAP, cells)")

	plt.subplot(222)
	xy = ds.attrs.UMAPVelocityPoints
	uv = ds.attrs.UMAPVelocity
	plt.scatter(ds.ca.UMAP[:, 0], ds.ca.UMAP[:, 1], c=colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=1, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (UMAP, grid)")

	plt.subplot(223)
	xy = ds.ca.TSNE
	uv = ds.ca.TSNEVelocity
	plt.scatter(xy[:, 0], xy[:, 1], c=colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=2.5, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (TSNE, cells)")

	plt.subplot(224)
	xy = ds.attrs.TSNEVelocityPoints
	uv = ds.attrs.TSNEVelocity
	plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=1, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (TSNE, grid)")

	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()
