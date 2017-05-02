from typing import *
import os
import csv
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi
import scipy.cluster.hierarchy as hierarchy
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc


class AggregateL2(luigi.Task):
	"""
	Aggregate all clusters in a new Loom file
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [
			cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class),
			cg.ClusterL2(tissue=self.tissue, major_class=self.major_class)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".agg.L2.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			ca_aggr = {
				"Age": "tally",
				"Clusters": "first",
				"Class": "first",
				"Class_Astrocyte": "mean",
				"Class_Cycling": "mean",
				"Class_Ependymal": "mean",
				"Class_Neurons": "mean",
				"Class_Immune": "mean",
				"Class_Oligos": "mean",
				"Class_OEC": "mean",
				"Class_Schwann": "mean",
				"Class_Vascular": "mean",
				"_Total": "mean",
				"Sex": "tally",
				"Tissue": "tally",
				"SampleID": "tally",
				"TissuePool": "first"
			}
			cells = ds.col_attrs["Clusters"] >= 0
			labels = ds.col_attrs["Clusters"][cells]
			n_labels = len(set(labels))

			logging.info("Aggregating clusters by mean")
			cg.aggregate_loom(ds, out_file, cells, "Clusters", "mean", ca_aggr)
			dsout = loompy.connect(out_file)

			logging.info("Trinarizing")
			trinaries = cg.Trinarizer().fit(ds)
			dsout.set_layer("trinaries", trinaries)

			logging.info("Computing cluster gene enrichment scores")
			(markers, enrichment) = cg.MarkerSelection(self.n_markers).fit(ds)
			dsout.set_layer("enrichment", enrichment)
			## TODO: make sure all marker genes are in this set
			## TODO: plot gene names for last cluster right-justified
			## TODO: redraw t-SNE using cluster ordering as initial layout (in circle)
			top_genes = np.argsort(np.max(enrichment, axis=1))[:1000]

			dsout.set_attr("NCells", np.bincount(labels, minlength=n_labels), axis=1)

			# Renumber the clusters
			data = np.log(dsout[sorted(top_genes), :] + 1)
			zx = hc.average(data.T)
			xordering = hc.leaves_list(zx)
			new_clusters = renumber(ds.col_attrs["Clusters"] + 1, np.arange(n_labels + 2), np.insert(xordering + 1, 0, 0)) - 1
			ds.set_attr("Clusters", new_clusters, axis=1)
			new_clusters = renumber(dsout.col_attrs["Clusters"], np.arange(n_labels + 1), xordering)
			dsout.set_attr("Clusters", new_clusters, axis=1)

			# Reorder the files by cluster ID
			logging.info("Permuting columns")
			ds.permute(np.argsort(ds.col_attrs["Clusters"]), axis=1)
			dsout.permute(np.argsort(dsout.col_attrs["Clusters"]), axis=1)

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


def renumber(a: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
	index = np.digitize(a.ravel(), keys, right=True)
	return(values[index].reshape(a.shape))

