import logging
from typing import Dict, List, Union

import numpy as np
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist

import loompy
from cytograph.annotation import AutoAnnotator, AutoAutoAnnotator
from cytograph.enrichment import FeatureSelectionByMultilevelEnrichment, Trinarizer
from cytograph.manifold import GraphSkeletonizer

from .config import load_config


class Aggregator:
	def __init__(self, *, f: Union[float, List[float]] = 0.2, mask: np.ndarray = None) -> None:
		self.f = f
		self.mask = mask

	def aggregate(self, ds: loompy.LoomConnection, *, out_file: str, agg_spec: Dict[str, str] = None) -> None:
		config = load_config()  # Generic config, just to get the paths
		if agg_spec is None:
			agg_spec = {
				"Age": "tally",
				"Clusters": "first",
				"Class": "mode",
				"Total": "mean",
				"Sex": "tally",
				"Tissue": "tally",
				"SampleID": "tally",
				"TissuePool": "first",
				"Outliers": "mean",
				"PCW": "mean"
			}
		cells = ds.col_attrs["Clusters"] >= 0
		labels = ds.col_attrs["Clusters"][cells]
		n_labels = len(set(labels))

		logging.info("Aggregating clusters")
		ds.aggregate(out_file, None, "Clusters", "mean", agg_spec)
		with loompy.connect(out_file) as dsout:

			if n_labels <= 1:
				return
				
			logging.info("Computing cluster gene enrichment scores")
			fe = FeatureSelectionByMultilevelEnrichment(mask=self.mask)
			markers = fe.fit(ds)
			dsout.layers["enrichment"] = fe.enrichment

			dsout.ca.NCells = np.bincount(labels, minlength=n_labels)

			# Renumber the clusters
			logging.info("Renumbering clusters by similarity, and permuting columns")

			data = np.log(dsout[:, :] + 1)[markers, :].T
			D = pdist(data, 'correlation')
			Z = hc.linkage(D, 'ward', optimal_ordering=True)
			ordering = hc.leaves_list(Z)

			# Permute the aggregated file, and renumber
			dsout.permute(ordering, axis=1)
			dsout.ca.Clusters = np.arange(n_labels)

			# Redo the Ward's linkage just to get a tree that corresponds with the new ordering
			data = np.log(dsout[:, :] + 1)[markers, :].T
			D = pdist(data, 'correlation')
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

			if n_labels > 3000:
				dsout.ca.MarkerGenes = np.empty(n_labels, dtype='str')
				dsout.ca.AutoAnnotation = np.empty(n_labels, dtype='str')
				return

			logging.info("Trinarizing")
			if type(self.f) is list or type(self.f) is tuple:
				for ix, f in enumerate(self.f):  # type: ignore
					trinaries = Trinarizer(f=f).fit(ds)
					if ix == 0:
						dsout.layers["trinaries"] = trinaries
					else:
						dsout.layers[f"trinaries_{f}"] = trinaries
			else:
				trinaries = Trinarizer(f=self.f).fit(ds)  # type:ignore
				dsout.layers["trinaries"] = trinaries

			logging.info("Computing auto-annotation")
			AutoAnnotator(root=config.paths.autoannotation, ds=dsout).annotate(dsout)

			logging.info("Computing auto-auto-annotation")
			AutoAutoAnnotator(n_genes=6).annotate(dsout)

			if "skeletonize" in config.steps:
				logging.info("Graph skeletonization")
				GraphSkeletonizer(min_pct=1).abstract(ds, dsout)
