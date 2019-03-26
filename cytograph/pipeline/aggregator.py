from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import scipy.cluster.hierarchy as hierarchy
import scipy
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
import scipy.stats
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from .utils import cc_genes_human, cc_genes_mouse, species
from .enrichment import MultilevelMarkerEnrichment


class Aggregator:
	def __init__(self, *, n_markers: int = 10, f: Union[float, List[float]] = 0.2, mask_cell_cycle: bool = True) -> None:
		self.n_markers = n_markers
		self.f = f
		self.mask_cell_cycle = mask_cell_cycle

	def aggregate(self, ds: loompy.LoomConnection, out_file: str, agg_spec: Dict[str, str] = None) -> None:
		cc_genes = cc_genes_human if species(ds) == "Homo sapiens" else cc_genes_mouse
		if agg_spec is None:
			agg_spec = {
				"Age": "tally",
				"Clusters": "first",
				"Class": "mode",
				"_Total": "mean",
				"Sex": "tally",
				"Tissue": "tally",
				"SampleID": "tally",
				"TissuePool": "first",
				"Outliers": "mean"
			}
		cells = ds.col_attrs["Clusters"] >= 0
		labels = ds.col_attrs["Clusters"][cells]
		n_labels = len(set(labels))

		logging.info("Aggregating clusters by mean")
		ds.aggregate(out_file, None, "Clusters", "mean", agg_spec)
		with loompy.connect(out_file) as dsout:
			logging.info("Trinarizing")
			if type(self.f) is list or type(self.f) is tuple:
				for ix, f in enumerate(self.f):  # type: ignore
					trinaries = cg.Trinarizer(f=f).fit(ds)
					if ix == 0:
						dsout.layers["trinaries"] = trinaries
					else:
						dsout.layers[f"trinaries_{f}"] = trinaries
			else:
				trinaries = cg.Trinarizer(f=self.f).fit(ds)
				dsout.layers["trinaries"] = trinaries

			logging.info("Computing cluster gene enrichment scores")
			mask = None
			if self.mask_cell_cycle:
				mask = np.isin(ds.ra.Gene, cc_genes)
			(markers, enrichment) = cg.MultilevelMarkerSelection(mask=mask).fit(ds)
			dsout.layers["enrichment"] = enrichment

			dsout.ca.NCells = np.bincount(labels, minlength=n_labels)

			# Renumber the clusters
			logging.info("Renumbering clusters by similarity, and permuting columns")

			data = np.log(dsout[:, :] + 1)[markers, :].T
			D = pdist(data, 'euclidean')
			Z = hc.linkage(D, 'ward', optimal_ordering=True)
			ordering = hc.leaves_list(Z)

			# Permute the aggregated file, and renumber
			dsout.permute(ordering, axis=1)
			dsout.ca.Clusters = np.arange(n_labels)

			# Redo the Ward's linkage just to get a tree that corresponds with the new ordering
			data = np.log(dsout[:, :] + 1)[markers, :].T
			D = pdist(data, 'euclidean')
			dsout.attrs.linkage = hc.linkage(D, 'ward', optimal_ordering=True)

			# Renumber the original file, and permute
			d = dict(zip(ordering, np.arange(n_labels)))
			new_clusters = np.array([d[x] if x in d else -1 for x in ds.ca.Clusters])
			ds.ca.Clusters = new_clusters
			ds.permute(np.argsort(ds.col_attrs["Clusters"]), axis=1)

			# Reorder the genes, markers first, ordered by enrichment in clusters
			logging.info("Permuting rows")
			mask = np.zeros(ds.shape[0], dtype=bool)
			mask[markers] = True
			# fetch enrichment from the aggregated file, so we get it already permuted on the column axis
			gene_order = np.zeros(ds.shape[0], dtype='int')
			gene_order[mask] = np.argmax(dsout.layer["enrichment"][mask, :], axis=1)
			gene_order[~mask] = np.argmax(dsout.layer["enrichment"][~mask, :], axis=1) + dsout.shape[1]
			gene_order = np.argsort(gene_order)
			ds.permute(gene_order, axis=0)
			dsout.permute(gene_order, axis=0)
