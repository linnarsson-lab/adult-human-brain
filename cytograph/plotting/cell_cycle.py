import matplotlib.pyplot as plt
import numpy as np

import loompy

from .colors import colorize


def cell_cycle(ds: loompy.LoomConnection, out_file: str) -> None:
	cc = ds.ca.CellCycle
	tsne = ds.ca.TSNE
	plt.figure(figsize=(12, 5))
	plt.subplot(121)
	plt.scatter(np.arange(cc.shape[0]), 100 * cc, marker=".", s=5)
	plt.ylim(0, 5)
	plt.hlines(1, 0, cc.shape[0], linestyles="dashed", colors="grey", lw=1)
	plt.xlabel("Cells")
	plt.ylabel("Cell cycle UMIs (%)")
	plt.title("Cell cycle index")
	plt.subplot(122)
	plt.scatter(tsne[:, 0], tsne[:, 1], c="lightgrey", marker=".", s=10)
	cells = cc > 0.01
	plt.scatter(tsne[cells, 0], tsne[cells, 1], vmin=0, vmax=0.05, c=cc[cells], marker=".", s=10)
	plt.axis("off")
	plt.title("Cycling cells (>1% cell cycle UMIs)")
	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()
