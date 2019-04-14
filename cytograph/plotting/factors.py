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


def factors(ds: loompy.LoomConnection, base_name: str) -> None:
	logging.info(f"Plotting factors")
	offset = 0
	theta = ds.ca.HPF
	beta = ds.ra.HPF
	if "HPFVelocity" in ds.ca:
		v_hpf = ds.ca.HPFVelocity
	n_factors = theta.shape[1]
	while offset < n_factors:
		fig = plt.figure(figsize=(15, 15))
		fig.subplots_adjust(hspace=0, wspace=0)
		for nnc in range(offset, offset + 8):
			if nnc >= n_factors:
				break
			ax = plt.subplot(4, 4, (nnc - offset) * 2 + 1)
			plt.xticks(())
			plt.yticks(())
			plt.axis("off")
			plt.scatter(x=ds.ca.TSNE[:, 0], y=ds.ca.TSNE[:, 1], c='lightgrey', marker='.', alpha=0.5, s=60, lw=0)
			cells = theta[:, nnc] > np.percentile(theta[:, nnc], 99) * 0.1
			cmap = "viridis"
			plt.scatter(x=ds.ca.TSNE[cells, 0], y=ds.ca.TSNE[cells, 1], c=theta[:, nnc][cells], marker='.', alpha=0.5, s=60, cmap=cmap, lw=0)
			ax.text(.01, .99, '\n'.join(ds.ra.Gene[np.argsort(-beta[:, nnc])][:9]), horizontalalignment='left', verticalalignment="top", transform=ax.transAxes)
			ax.text(.99, .9, f"{nnc}", horizontalalignment='right', transform=ax.transAxes, fontsize=12)
			if "HPFVelocity" in ds.ca:
				ax.text(.5, .9, f"{np.percentile(v_hpf[nnc], 98):.2}", horizontalalignment='right', transform=ax.transAxes)
				vcmap = LinearSegmentedColormap.from_list("", ["red","whitesmoke","green"])
				norm = MidpointNormalize(midpoint=0)
				ax = plt.subplot(4, 4, (nnc - offset) * 2 + 2)
				plt.scatter(ds.ca.TSNE[:,0], ds.ca.TSNE[:,1],vmin=np.percentile(v_hpf[:,nnc], 2),vmax=np.percentile(v_hpf[:,nnc], 98), c=v_hpf[:,nnc],norm=norm, cmap=vcmap, marker='.',s=10)
				plt.axis("off")
		plt.savefig(base_name + f"{offset}.png", dpi=144)
		offset += 8
		plt.close()
