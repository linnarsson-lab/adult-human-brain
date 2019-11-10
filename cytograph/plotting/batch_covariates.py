import matplotlib.pyplot as plt
import numpy as np

import loompy

from .colors import colorize


def batch_covariates(ds: loompy.LoomConnection, out_file: str) -> None:
	xy = ds.ca.TSNE
	plt.figure(figsize=(12, 12))

	if "Tissue" in ds.ca:
		names, labels = np.unique(ds.ca.Tissue, return_inverse=True)
	else:
		names, labels = np.unique(np.array(["(unknown)"] * ds.shape[1]), return_inverse=True)
	ax = plt.subplot(221)
	colors = colorize(names)
	cells = np.random.permutation(labels.shape[0])
	ax.scatter(xy[cells, 0], xy[cells, 1], c=colors[labels][cells], lw=0, s=10)
	h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")
	ax.legend(
		handles=[h(colors[i]) for i in range(len(names))],
		labels=list(names),
		loc='lower left',
		markerscale=1,
		frameon=False,
		fontsize=10)
	plt.title("Tissue")

	if "PCW" in ds.ca:
		names, labels = np.unique(ds.ca.PCW, return_inverse=True)
	elif "Age" in ds.ca:
		names, labels = np.unique(ds.ca.Age, return_inverse=True)
	else:
		labels = np.array(["(unknown)"] * ds.shape[1])
	ax = plt.subplot(222)
	colors = colorize(names)
	cells = np.random.permutation(labels.shape[0])
	ax.scatter(xy[cells, 0], xy[cells, 1], c=colors[labels][cells], lw=0, s=10)
	h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")
	ax.legend(
		handles=[h(colors[i]) for i in range(len(names))],
		labels=list(names),
		loc='upper right',
		markerscale=1,
		frameon=False,
		fontsize=10)
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
		names, labels = np.unique(ds.ca.SampleID, return_inverse=True)
	else:
		names, labels = np.unique(np.array(["(unknown)"] * ds.shape[1]), return_inverse=True)
	ax = plt.subplot(224)
	colors = colorize(names)
	cells = np.random.permutation(labels.shape[0])
	ax.scatter(xy[cells, 0], xy[cells, 1], c=colors[labels][cells], lw=0, s=10)
	h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")
	ax.legend(
		handles=[h(colors[i]) for i in range(len(names))],
		labels=list(names),
		loc='upper right',
		markerscale=1,
		frameon=False,
		fontsize=10)
	plt.title("SampleID")

	plt.savefig(out_file, dpi=144)
	plt.close()
