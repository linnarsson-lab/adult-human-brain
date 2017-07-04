from typing import *
import os
import logging
import loompy
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi


class ExportL2(luigi.Task):
	"""
	Luigi Task to export summary files
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [
			cg.AggregateL2(tissue=self.tissue, major_class=self.major_class),
			cg.ClusterL2(tissue=self.tissue, major_class=self.major_class)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L2_" + self.major_class + "_" + self.tissue + "_exported"))

	def run(self) -> None:
		with self.output().temporary_path() as out_dir:
			logging.info("Exporting cluster data")
			if not os.path.exists(out_dir):
				os.mkdir(out_dir)
			dsagg = loompy.connect(self.input()[0].fn)
			dsagg.export(os.path.join(out_dir, "L2_" + self.major_class + self.tissue + "_expression.tab"))
			dsagg.export(os.path.join(out_dir, "L2_" + self.major_class + self.tissue + "_enrichment.tab"), layer="enrichment")
			dsagg.export(os.path.join(out_dir, "L2_" + self.major_class + self.tissue + "_enrichment_q.tab"), layer="enrichment_q")
			dsagg.export(os.path.join(out_dir, "L2_" + self.major_class + self.tissue + "_trinaries.tab"), layer="trinaries")

			logging.info("Plotting manifold graph with auto-annotation")
			tags = list(dsagg.col_attrs["AutoAnnotation"])
			ds = loompy.connect(self.input()[1].fn)
			cg.plot_graph(ds, os.path.join(out_dir, "L2_" + self.major_class + self.tissue + "_manifold.aa.png"), tags)

			logging.info("Plotting manifold graph with auto-auto-annotation")
			tags = list(dsagg.col_attrs["MarkerGenes"])
			cg.plot_graph(ds, os.path.join(out_dir, "L2_" + self.major_class + self.tissue + "_manifold.aaa.png"), tags)

			logging.info("Plotting marker heatmap")
			cg.plot_markerheatmap(ds, dsagg, n_markers_per_cluster=self.n_markers, out_file=os.path.join(out_dir, "L2_" + self.major_class + self.tissue + "_heatmap.pdf"))
