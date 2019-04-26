import matplotlib.pyplot as plt
import numpy as np

import loompy

from .colors import colorize


def cell_cycle(ds: loompy.LoomConnection, out_file: str) -> None:
	(g1, s, g2m) = ds.ca.CellCycle_G1, ds.ca.CellCycle_S, ds.ca.CellCycle_G2M
	ordering = np.random.permutation(ds.shape[1])
	tsne_x = ds.ca.TSNE[:, 0][ordering]
	tsne_y = ds.ca.TSNE[:, 1][ordering]
	g1 = g1[ordering]
	s = s[ordering]
	g2m = g2m[ordering]
	colors = colorize(ds.ca.Clusters)[ordering]

	plt.figure(figsize=(20, 4))
	plt.subplot(141)
	cells = (g1 + s + g2m) > np.percentile((g1 + s + g2m), 99) / 5
	plt.scatter(tsne_x, tsne_y, c='lightgrey', marker='.', lw=0)
	plt.scatter(tsne_x[cells], tsne_y[cells], c=colors[cells], marker='.', lw=0)
	plt.title("Cells at >20% of 99th percentile")
	plt.subplot(142)
	cells = g1 > np.percentile(g1, 99) / 5
	plt.scatter(tsne_x, tsne_y, c='lightgrey', marker='.', lw=0)
	plt.scatter(tsne_x[cells], tsne_y[cells], c=g1[cells], marker='.', lw=0, s=30, alpha=0.7)
	plt.title("G1")
	plt.subplot(143)
	cells = s > np.percentile(s, 99) / 5
	plt.scatter(tsne_x, tsne_y, c='lightgrey', marker='.', lw=0)
	plt.scatter(x=tsne_x[cells], y=tsne_y[cells], c=s[cells], marker='.', lw=0, s=30, alpha=0.7)
	plt.title("S")
	plt.subplot(144)
	cells = g2m > np.percentile(g2m, 99) / 5
	plt.scatter(tsne_x, tsne_y, c='lightgrey', marker='.', lw=0)
	plt.scatter(tsne_x[cells], tsne_y[cells], c=g2m[cells], marker='.', lw=0, s=30, alpha=0.7)
	plt.title("G2/M")
	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()
