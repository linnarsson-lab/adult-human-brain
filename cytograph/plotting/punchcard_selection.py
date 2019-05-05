import matplotlib.pyplot as plt
import numpy as np
import loompy
from .colors import colorize, colors75


def punchcard_selection(ds: loompy.LoomConnection, out_file: str = None) -> None:
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111)
	pos = ds.ca.TSNE

	# Plotting TSNE colored by subset
	subsets = np.unique(ds.ca.Subset)
	colors = colorize(subsets)
	subset_colors = {subsets[i]: colors[i] for i in range(len(subsets))}
	for subset in np.unique(ds.ca.Subset):
		cells = ds.ca.Subset == subset
		plt.scatter(pos[cells, 0], pos[cells, 1], c=("lightgrey" if subset == "" else subset_colors[subset]), label=subset, lw=0, marker='.', s=40, alpha=0.5)
		plt.axis("off")
	plt.legend(markerscale=3, loc="upper right")

	# Labeling clusters
	labels = ds.ca.Clusters
	for lbl in range(0, max(labels) + 1):
		txt = str(lbl)
		(x, y) = np.median(pos[np.where(labels == lbl)[0]], axis=0)
		subset = ds.ca.Subset[labels == lbl][0]
		assert np.all(ds.ca.Subset[labels == lbl] == subset)
		ax.text(x, y, txt, fontsize=12, bbox=dict(facecolor=subset_colors[subset], alpha=0.5, ec='none'))

	plt.title("Subsets selected by punchcard")

	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()
