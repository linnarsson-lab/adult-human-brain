from typing import *
import os
import logging
import loompy
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi


class ExportAnalysis(luigi.Task):
	"""
	Luigi Task to export summary files
	"""
	analysis = luigi.Parameter()
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [
			cg.AggregateAnalysis(analysis=self.analysis),
			cg.ClusterAnalysis(analysis=self.analysis)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "Analysis_" + self.analysis + "_exported"))

	def run(self) -> None:
		logging.info("Exporting cluster data")
		with self.output().temporary_path() as out_dir:
			if not os.path.exists(out_dir):
				os.mkdir(out_dir)
			dsagg = loompy.connect(self.input()[0].fn)
			dsagg.export(os.path.join(out_dir, "Analysis_" + self.analysis + "_expression.tab"))
			dsagg.export(os.path.join(out_dir, "Analysis_" + self.analysis + "_enrichment.tab"), layer="enrichment")
			dsagg.export(os.path.join(out_dir, "Analysis_" + self.analysis + "_enrichment_q.tab"), layer="enrichment_q")
			dsagg.export(os.path.join(out_dir, "Analysis_" + self.analysis + "_trinaries.tab"), layer="trinaries")

			ds = loompy.connect(self.input()[1].fn)

			logging.info("Plotting manifold graph with auto-annotation")
			tags = list(dsagg.col_attrs["AutoAnnotation"])
			cg.plot_graph(ds, os.path.join(out_dir, "Analysis_" + self.analysis + "_manifold.aa.png"), tags)

			logging.info("Plotting manifold graph with auto-auto-annotation")
			tags = list(dsagg.col_attrs["MarkerGenes"])
			cg.plot_graph(ds, os.path.join(out_dir, "Analysis_" + self.analysis + "_manifold.aaa.png"), tags)

			logging.info("Plotting marker heatmap")
			cg.plot_markerheatmap(ds, dsagg, n_markers_per_cluster=self.n_markers, out_file=os.path.join(out_dir, "Analysis_" + self.analysis + "_heatmap.pdf"))
