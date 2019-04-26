import matplotlib.pyplot as plt
import numpy as np

import loompy

from .colors import colorize, colors75


def buckets(ds: loompy.LoomConnection, out_file: str = None) -> None:
	plt.figure(figsize=(21, 7))
	plt.subplot(131)
	buckets = np.unique(ds.ca.Bucket)
	colors = colorize(buckets)
	bucket_colors = {buckets[i]: colors[i] for i in range(len(buckets))}
	for ix, bucket in enumerate(np.unique(ds.ca.Bucket)):
		cells = ds.ca.Bucket == bucket
		plt.scatter(ds.ca.TSNE[cells, 0], ds.ca.TSNE[cells, 1], c=("lightgrey" if bucket == "" else colors[ix]), label=bucket, lw=0, marker='.', s=40, alpha=0.5)
		plt.axis("off")
	plt.legend(markerscale=3, loc="upper right")
	plt.title("Buckets from previous build")
	plt.subplot(132)
	cells = ds.ca.NewCells == 1
	plt.scatter(ds.ca.TSNE[~cells, 0], ds.ca.TSNE[~cells, 1], c="lightgrey", label="Old cells", lw=0, marker='.', s=40, alpha=0.5)
	plt.scatter(ds.ca.TSNE[cells, 0], ds.ca.TSNE[cells, 1], c="red", label="New cells", lw=0, marker='.', s=40, alpha=0.5)
	plt.axis("off")
	plt.legend(markerscale=3, loc="upper right")
	plt.title("Cells added in this build")
	plt.subplot(133)
	n_colors = len(bucket_colors)
	for ix, bucket in enumerate(np.unique(ds.ca.NewBucket)):
		cells = ds.ca.NewBucket == bucket
		color = "lightgrey"
		if bucket != "":
			if bucket in bucket_colors.keys():
				color = bucket_colors[bucket]
			else:
				color = colors75[n_colors]
				bucket_colors[bucket] = color
				n_colors += 1
		plt.scatter(ds.ca.TSNE[cells, 0], ds.ca.TSNE[cells, 1], c=color, label=bucket, lw=0, marker='.', s=40, alpha=0.5)
		plt.axis("off")
	plt.legend(markerscale=3, loc="upper right")
	plt.title("Buckets proposed for this build")

	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()
