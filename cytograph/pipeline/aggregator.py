from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import scipy.cluster.hierarchy as hierarchy
import scipy
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
import scipy.stats
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
from cytograph.species import species
from cytograph.enrichment import Trinarizer, FeatureSelectionByMultilevelEnrichment
from cytograph.annotation import AutoAnnotator, AutoAutoAnnotator
from cytograph.clustering import ClusterValidator
import cytograph.plotting as cgplot
from .utils import Tempname
from .config import config


class Aggregator:
	def __init__(self, *, f: Union[float, List[float]] = 0.2, mask: np.ndarray = None) -> None:
		self.f = f
		self.mask = mask

	def aggregate(self, ds: loompy.LoomConnection, *, agg_file: str, export_dir: str = None, agg_spec: Dict[str, str] = None) -> None:
		if os.path.exists(agg_file):
			logging.info(f"Skipping aggregation of {agg_file} because file already exists")

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
		with Tempname(agg_file) as out_file:
			ds.aggregate(out_file, None, "Clusters", "mean", agg_spec)
			with loompy.connect(out_file) as dsout:
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

				logging.info("Computing cluster gene enrichment scores")
				fe = FeatureSelectionByMultilevelEnrichment(mask=self.mask)
				markers = fe.fit(ds)
				dsout.layers["enrichment"] = fe.enrichment

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

				logging.info("Computing auto-annotation")
				AutoAnnotator(root=config.paths.autoannotation).annotate(dsout)

				logging.info("Computing auto-auto-annotation")
				AutoAutoAnnotator(n_genes=6).annotate(dsout)

				if export_dir is not None and not os.path.exists(export_dir):
					pool = os.path.basename(agg_file[:-9])
					logging.info(f"Exporting plots to {export_dir}")
					with Tempname(export_dir) as out_dir:
						os.mkdir(out_dir)
						cgplot.manifold(ds, os.path.join(out_dir, f"{pool}_TSNE_manifold.aa.png"), list(dsout.ca.AutoAnnotation))
						cgplot.manifold(ds, os.path.join(out_dir, pool + "_TSNE_manifold.aaa.png"), list(dsout.ca.MarkerGenes))
						cgplot.manifold(ds, os.path.join(out_dir, pool + "_UMAP_manifold.aaa.png"), list(dsout.ca.MarkerGenes), embedding="UMAP")
						cgplot.markerheatmap(ds, dsout, n_markers_per_cluster=10, out_file=os.path.join(out_dir, pool + "_heatmap.pdf"))
						cgplot.factors(ds, base_name=os.path.join(out_dir, pool + "_factors"))
						cgplot.cell_cycle(ds, os.path.join(out_dir, pool + "_cellcycle.png"))
						cgplot.radius_characteristics(ds, out_file=os.path.join(out_dir, pool + "_neighborhoods.png"))
						cgplot.batch_covariates(ds, out_file=os.path.join(out_dir, pool + "_batches.png"))
						cgplot.umi_genes(ds, out_file=os.path.join(out_dir, pool + "_umi_genes.png"))
						ClusterValidator().fit(ds, os.path.join(out_dir, f"{pool}_cluster_pp.png"))
						cgplot.embedded_velocity(ds, out_file=os.path.join(out_dir, f"{pool}_velocity.png"))
						cgplot.TFs(ds, dsout, out_file_root=os.path.join(out_dir, pool))
