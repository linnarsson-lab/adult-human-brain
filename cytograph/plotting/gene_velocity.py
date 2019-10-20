import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import loompy

from .colors import colorize
from .midpoint_normalize import MidpointNormalize


def gene_velocity(ds: loompy.LoomConnection, gene: str, out_file: str = None) -> None:
	g = ds.gamma[ds.ra.Gene == gene]
	s = ds["spliced_exp"][ds.ra.Gene == gene, :][0]
	u = ds["unspliced_exp"][ds.ra.Gene == gene, :][0]
	v = ds["velocity"][ds.ra.Gene == gene, :][0]
	c = ds.ca.Clusters
	vcmap = LinearSegmentedColormap.from_list("", ["red", "whitesmoke", "green"])

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
	plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=s, cmap="viridis", marker='.', s=10)
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
