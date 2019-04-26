import matplotlib.pyplot as plt

import loompy

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
	plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Clusters), lw=0, s=15, alpha=0.1)
	plt.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], angles='xy', scale_units='xy', scale=1, edgecolor='white', facecolor="black", linewidth=.25)
	plt.axis("off")
	plt.title("Velocity (TSNE, grid)")

	if out_file is not None:
		plt.savefig(out_file, dpi=144)
	plt.close()
