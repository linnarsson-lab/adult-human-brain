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


def batch_covariates(ds: loompy.LoomConnection, out_file: str) -> None:
	xy = ds.ca.TSNE
	plt.figure(figsize=(12, 12))

	if "Tissue" in ds.ca:
		labels = ds.ca.Tissue
	else:
		labels = np.array(["(unknown)"] * ds.shape[1])
	ax = plt.subplot(221)
	for lbl in np.unique(labels):
		cells = labels == lbl
		ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=colorize(labels)[cells], label=lbl, lw=0, s=10)
	ax.legend()
	plt.title("Tissue")

	if "PCW" in ds.ca:
		labels = ds.ca.PCW
	elif "Age" in ds.ca:
		labels = ds.ca.Age
	else:
		labels = np.array(["(unknown)"] * ds.shape[1])
	ax = plt.subplot(222)
	for lbl in np.unique(labels):
		cells = labels == lbl
		ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=colorize(labels)[cells], label=lbl, lw=0, s=0)
	cells = np.random.permutation(labels.shape[0])
	ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=colorize(labels)[cells], lw=0, s=10)
	lgnd = ax.legend()
	for handle in lgnd.legendHandles:
		handle.set_sizes([10])
	plt.title("Age")
	
	if "XIST" in ds.ra.Gene:
		xist = ds[ds.ra.Gene == "XIST", :][0]
	elif "Xist" in ds.ra.Gene:
		xist = ds[ds.ra.Gene == "Xist", :][0]
	else:
		xist = np.array([0] * ds.shape[1])
	ax = plt.subplot(223)
	cells = xist > 0
	ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', lw=0, s=10)
	ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=xist[cells], lw=0, s=10)
	plt.title("Sex (XIST)")

	if "SampleID" in ds.ca:
		labels = ds.ca.SampleID
	else:
		labels = np.array(["(unknown)"] * ds.shape[1])
	ax = plt.subplot(224)
	for lbl in np.unique(labels):
		cells = labels == lbl
		ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=colorize(labels)[cells], label=lbl, lw=0, s=10)
	ax.legend()
	plt.title("SampleID")

	plt.savefig(out_file, dpi=144)
	plt.close()
