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


class AggregateClustersL3(luigi.Task):
	"""
	Aggregate all clusters in a new Loom file
	"""

	def requires(self) -> Iterator[luigi.Task]:
		return cg.PoolAllL3()

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "Adolescent.L3.aggregated.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
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

			logging.info("Aggregating clusters by geometric mean")
			cg.aggregate_loom(ds, out_file, cells, "Clusters", "geom", ca_aggr)
			dsout = loompy.connect(out_file)

			logging.info("Trinarizing all clusters")
			trinaries = cg.Trinarizer().fit(ds)
			dsout.set_layer("trinaries", trinaries)

			logging.info("Computing cluster gene enrichment scores")
			(markers, enrichment) = cg.MarkerSelection(2).fit(ds)
			dsout.set_layer("enrichment", enrichment)

			dsout.set_attr("NCells", np.bincount(labels, minlength=n_labels), axis=1)

			best_markers = np.zeros(dsout.shape[0], dtype='int')
			best_markers[markers] = 1
			dsout.set_attr("BestMarkers", best_markers, axis=0)
