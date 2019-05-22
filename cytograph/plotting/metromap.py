import numpy as np
import loompy
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .colors import tube_colors


def renumber_components(labels: np.ndarray) -> np.ndarray:
	# Set all the singletons to -1
	for i in np.unique(labels):
		if (i == labels).sum() == 1:
			labels[labels == i] = -1
	# Renumber the labels
	ix = 0
	while ix <= labels.max():
		# Skip past occupied labels
		if (labels == ix).sum() > 0:
			ix += 1
			continue
		# Now we're at a free slot
		j = ix
		while (labels == j).sum() == 0:
			j += 1
		labels[labels == j] = ix
	return labels


def metromap(ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, out_file: str = None, embedding: str = "TSNE") -> None:
	ga = dsagg.col_graphs.GA
	n_components, labels = sparse.csgraph.connected_components(csgraph=ga, directed=False, return_labels=True)
	labels = renumber_components(labels)
	aspect_ratio = (ds.ca[embedding][:, 0].max() - ds.ca[embedding][:, 0].min()) / (ds.ca[embedding][:, 1].max() - ds.ca[embedding][:, 1].min())
	plt.figure(figsize=(10 * aspect_ratio, 10))
	ax = plt.subplot(111)
	plt.scatter(ds.ca[embedding][:, 0], ds.ca[embedding][:, 1], s=10, lw=0, c="lightgrey")
	r = ((ds.ca[embedding][:, 0].max() - ds.ca[embedding][:, 0].min()) + (ds.ca[embedding][:, 1].max() - ds.ca[embedding][:, 1].min())) / 100
	for c in np.unique(ds.ca.Clusters):
		ls = ":" if labels[c] == -1 else "-"
		ax.add_artist(plt.Circle((dsagg.ca[embedding][c, 0], dsagg.ca[embedding][c, 1]), lw=2, radius=r, fill=True, linestyle=ls, fc="white", ec="black", alpha=0.8))
		ax.add_artist(plt.Text(dsagg.ca[embedding][c, 0], dsagg.ca[embedding][c, 1], str(c), fontname="Verdana", fontsize=11, zorder=1, va="center", ha="center"))
	lc = LineCollection(zip(dsagg.ca[embedding][ga.row], dsagg.ca[embedding][ga.col]), linewidths=6, color=tube_colors[labels[ga.row]], alpha=1, zorder=1)
	ax.add_collection(lc)
	plt.axis("off")
	if out_file is not None:
		plt.savefig(out_file, dpi=144)
		plt.close()
