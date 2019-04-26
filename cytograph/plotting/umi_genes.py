import matplotlib.pyplot as plt
import numpy as np
import loompy


def umi_genes(ds: loompy.LoomConnection, out_file: str) -> None:
	plt.figure(figsize=(16, 4))
	plt.subplot(131)
	for chip in np.unique(ds.ca.SampleID):
		cells = ds.ca.SampleID == chip
		plt.hist(ds.ca.TotalUMI[cells], bins=100, label=chip, alpha=0.5, range=(0, 30000))
		plt.title("UMI distribution")
		plt.ylabel("Number of cells")
		plt.xlabel("Number of UMIs")
	plt.legend()
	plt.subplot(132)
	for chip in np.unique(ds.ca.SampleID):
		cells = ds.ca.SampleID == chip
		plt.hist(ds.ca.NGenes[cells], bins=100, label=chip, alpha=0.5, range=(0, 10000))
		plt.title("Gene count distribution")
		plt.ylabel("Number of cells")
		plt.xlabel("Number of genes")
	plt.legend()
	plt.subplot(133)
	tsne = ds.ca.TSNE
	plt.scatter(tsne[:, 0], tsne[:, 1], c="lightgrey", lw=0, marker='.')
	for chip in np.unique(ds.ca.SampleID):
		cells = (ds.ca.ScrubletFlag == 1) & (ds.ca.SampleID == chip)
		plt.scatter(tsne[:, 0][cells], tsne[:, 1][cells], label=chip, lw=0, marker='.')
		plt.title("Doublets")
	plt.axis("off")
	plt.legend()
	plt.savefig(out_file, dpi=144)
	plt.close()
