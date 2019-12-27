from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import loompy
from cytograph.species import Species

from .colors import colorize
from .dendrogram import dendrogram


class Heatmap():
	def __init__(self, genes: np.ndarray, attrs: Dict[str, str], markers: Dict[str, List[str]] = None, layer: str = "pooled") -> None:
		self.layer = layer
		self.genes = genes
		self.markers = markers
		self.attrs = attrs
	
	def plot(self, ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, out_file: str = None) -> None:
		layer = self.layer if self.layer in ds.layers else ""
		if self.markers is None:
			self.markers = Species.detect(ds).markers
		
		n_clusters = np.max(dsagg.ca.Clusters) + 1
		if np.all(ds.ra.Gene[self.genes] == ds.ra.Gene[:len(self.genes)]):
			data = np.log(ds[layer][:len(self.genes), :] + 1)
			enrichment = dsagg.layer["enrichment"][:len(self.genes), :]
		else:
			data = np.log(ds[layer][self.genes, :] + 1)
			enrichment = dsagg.layer["enrichment"][:len(self.genes), :]
	
		# Order the dataset by cluster enrichment
		top_cluster = []
		for g in ds.ra.Gene[self.genes]:
			top_cluster.append(np.argsort(-dsagg["enrichment"][ds.ra.Gene == g, :][0])[0])
		ordering = np.argsort(top_cluster)
		data = data[ordering, :]
		enrichment = enrichment[ordering, :]
		gene_names = ds.ra.Gene[self.genes][ordering]

		clusterborders = np.cumsum(dsagg.ca.NCells)
		clustermiddles = clusterborders[:-1] + (clusterborders[1:] - clusterborders[:-1]) / 2
		clustermiddles = np.hstack([[clusterborders[0]/2], clustermiddles])  # Add the first cluster
		gene_pos = clusterborders[np.array(top_cluster)[ordering]]

		# Calculate the plot height
		num_strips = 0
		for attr, kind in self.attrs.items():
			if attr not in ds.ca:
				continue
			if kind == "ticker":
				num_strips += np.unique(ds.ca[attr]).shape[0]
			else:
				num_strips += 1
		num_markers = sum([len(g) for g in self.markers.values()])
		# Height in terms of heatmap rows
		dendr_height = 10 if "linkage" in dsagg.attrs else 0
		strip_height = 2
		total_height = data.shape[0] + strip_height * (num_strips + num_markers) + dendr_height
		color_range = np.percentile(data, 99, axis=1) + 0.1
		data_scaled = data / color_range[None].T

		plt.figure(figsize=(12, total_height / 10))
		grid = (total_height, 1)
		offset = 0  # Start at top with the dendrogram

		if "linkage" in dsagg.attrs:
			ax = plt.subplot2grid(grid, (offset, 0), rowspan=dendr_height)
			offset += dendr_height
			lc = dendrogram(dsagg.attrs.linkage, leaf_positions=clustermiddles)
			ax.add_collection(lc)
			plt.xlim(0, clusterborders[-1])
			plt.ylim(0, dsagg.attrs.linkage[:, 2].max() * 1.1)
			plt.axis("off")

		for attr, spec in self.attrs.items():
			if attr not in ds.ca:
				continue
			if ":" in spec:
				kind, transform = spec.split(":")
			else:
				kind = spec
				transform = ""
			if kind == "ticker":
				for val in np.unique(ds.ca[attr]):
					d = (ds.ca[attr] == val).astype("int")
					ax = plt.subplot2grid(grid, (offset, 0), rowspan=strip_height)
					offset += strip_height
					plt.imshow(np.expand_dims(d, axis=0), aspect='auto', cmap="Greys")
					plt.text(0, 0.9, val, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="black")
					plt.axis("off")
				plt.text(1, 0.9, attr, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="black")
			elif kind == "categorical":
				d = colorize(np.nan_to_num(ds.ca[attr]))
				ax = plt.subplot2grid(grid, (offset, 0), rowspan=strip_height)
				offset += strip_height
				plt.imshow(np.expand_dims(d, axis=0), aspect='auto')
				plt.text(0, 0.9, attr, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="black")
			else:
				d = ds.ca[attr]
				if transform == "log":
					d = np.log(d + 1)
				ax = plt.subplot2grid(grid, (offset, 0), rowspan=strip_height)
				offset += strip_height
				plt.imshow(np.expand_dims(d, axis=0), aspect='auto', cmap=kind)
				plt.text(0, 0.9, attr, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="black")
			plt.axis("off")

		for cat, markers in self.markers.items():
			write_cat = True
			for m in markers:
				if m not in ds.ra.Gene:
					vals = np.zeros(ds.shape[1])
				else:
					vals = ds[layer][ds.ra.Gene == m, :][0]
				vals = vals / (np.percentile(vals, 99) + 0.1)
				ax = plt.subplot2grid(grid, (offset, 0), rowspan=strip_height)
				offset += strip_height
				ax.imshow(np.expand_dims(vals, axis=0), aspect='auto', cmap="viridis", vmin=0, vmax=1)
				if write_cat:
					plt.text(1.001, 0.9, cat, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="black")
					write_cat = False
				plt.text(0, 0.9, m, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="black")
				plt.text(0.5, 0.9, m, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white")
				plt.text(0.999, 0.9, m, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=7, color="white")
				plt.axis("off")

		ax = plt.subplot2grid(grid, (offset, 0), rowspan=data.shape[0])

		# Draw border between clusters
		if n_clusters > 2:
			tops = np.vstack((clusterborders - 0.5, np.zeros(clusterborders.shape[0]) - 0.5)).T
			bottoms = np.vstack((clusterborders - 0.5, np.zeros(clusterborders.shape[0]) + data.shape[0] - 0.5)).T
			lc = LineCollection(zip(tops, bottoms), linewidths=1, color='white', alpha=0.5)
			ax.add_collection(lc)

		ax.imshow(data_scaled, aspect='auto', cmap="viridis", vmin=0, vmax=1)
		n_genes = ds.ra.Gene[self.genes].shape[0]
		for ix, gene in enumerate(gene_names):
			xpos = gene_pos[ix]
			if xpos == clusterborders[-1]:
				if n_clusters > 2:
					xpos = clusterborders[-3]
			plt.text(0.001 + xpos, ix - 0.5, gene, horizontalalignment='left', verticalalignment='top', fontsize=4, color="white")
			plt.text(0, 1 - ix / n_genes, gene, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=4, color="black")
			plt.text(1, 1 - ix / n_genes, gene, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=4, color="black")

		# Cluster IDs
		labels = [str(x) for x in np.arange(n_clusters)]
		if "ClusterName" in dsagg.ca:
			labels = dsagg.ca.ClusterName
		for ix, x in enumerate(clustermiddles):
			plt.text(x, np.mod(ix, 4), labels[ix], horizontalalignment='center', verticalalignment='top', fontsize=6, color="white", weight="bold")
			plt.text(x, np.mod(ix, 4) + n_genes / 2, labels[ix], horizontalalignment='center', verticalalignment='top', fontsize=6, color="white", weight="bold")
			plt.text(x, np.mod(ix, 4) + n_genes - 5, labels[ix], horizontalalignment='center', verticalalignment='top', fontsize=6, color="white", weight="bold")

		ax.set_frame_on(False)
		ax.set_xticks([])
		ax.set_yticks([])

		plt.subplots_adjust(hspace=0)
		if out_file is not None:
			plt.savefig(out_file, format="pdf", dpi=144, bbox_inches='tight')
			plt.close()
