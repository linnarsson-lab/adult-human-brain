import logging
from typing import Dict, List, Union
import numpy as np
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from sklearn.preprocessing import scale
import loompy
from cytograph.annotation import AutoAnnotator, AutoAutoAnnotator
from cytograph.enrichment import Enrichment, Trinarizer
from cytograph.manifold import GraphSkeletonizer
from .config import Config


class Aggregator:
	"""
	Generates a new loom file aggregated by cluster
	"""

	def __init__(self, *, config: Config, f: Union[float, List[float]] = 0.2, mask: np.ndarray = None) -> None:
		self.f = f
		self.mask = mask
		self.config = config

	def aggregate(self, ds: loompy.LoomConnection, *, out_file: str, agg_spec: Dict[str, str] = None) -> None:
		"""
		Generates a new loom file aggregated by cluster
		"""
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

			dsout.ca.NCells = np.bincount(labels, minlength=n_labels)

			logging.info("Computing gene enrichment and nonzero fractions")
			enr = Enrichment()
			dsout.layers["enrichment"] = enr.fit(dsout, ds)
			dsout.layers["nonzeros"] = enr.nonzeros

			# Renumber the clusters
			logging.info("Renumbering clusters by similarity, and permuting columns")
			markers = ds.ra.Selected == 1
			total_genes = np.sum(dsout[:, :], axis=0)
			data = np.log(dsout[:, :] / total_genes * np.median(total_genes) + 1)[markers, :].T
			data = scale(data)
			D = pdist(data, 'euclidean')
			Z = hc.linkage(D, 'ward', optimal_ordering=True)
			ordering = hc.leaves_list(Z)

			# Permute the aggregated file, and renumber
			dsout.permute(ordering, axis=1)
			dsout.ca.Clusters = np.arange(n_labels)

			# Redo the Ward's linkage just to get a tree that corresponds with the new ordering
			data = np.log(dsout[:, :] / total_genes * np.median(total_genes) + 1)[markers, :].T
			data = scale(data)
			D = pdist(data, 'euclidean')
			dsout.attrs.linkage = hc.linkage(D, 'ward', optimal_ordering=True)

			# Renumber the original file, and permute
			d = dict(zip(ordering, np.arange(n_labels)))
			new_clusters = np.array([d[x] if x in d else -1 for x in ds.ca.Clusters])
			ds.ca.Clusters = new_clusters
			ds.permute(np.argsort(ds.col_attrs["Clusters"]), axis=1)

			# Find cluster markers
			n_markers = 10
			if self.mask is None:
				excluded = set(np.where(ds.ra.Valid == 0)[0])
			else:
				excluded = set(np.where(np.logical_or(ds.ra.Valid == 0, self.mask))[0])

			included = []
			for ix in range(n_labels):
				enriched = np.argsort(dsout.layers["enrichment"][:, ix])[::-1]
				n = 0
				count = 0
				while count < n_markers:
					if enriched[n] in excluded:
						n += 1
						continue
					included.append(enriched[n])
					excluded.add(enriched[n])
					n += 1
					count += 1
			markers = np.array(included)

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

			logging.info("Trinarizing")
			trinaries = Trinarizer(f=self.f).fit(dsout)
			dsout.layers["trinaries"] = trinaries

			logging.info("Computing auto-annotation")
			AutoAnnotator(root=self.config.paths.autoannotation, ds=dsout).annotate(dsout)

			logging.info("Computing auto-auto-annotation")
			AutoAutoAnnotator(n_genes=6).annotate(dsout)

			if "skeletonize" in self.config.steps:
				logging.info("Graph skeletonization")
				GraphSkeletonizer(min_pct=1).abstract(ds, dsout)
