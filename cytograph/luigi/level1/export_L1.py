from typing import *
import os
#import logging
import loompy
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi


class ExportL1(luigi.Task):
	"""
	Luigi Task to export summary files
	"""
	tissue = luigi.Parameter()
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [
			cg.AggregateL1(tissue=self.tissue),
			cg.ClusterL1(tissue=self.tissue)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L1_" + self.tissue + "_exported"))

	def run(self) -> None:
		logging = cg.logging(self, True)
		logging.info("Exporting cluster data")
		with self.output().temporary_path() as out_dir:
			if not os.path.exists(out_dir):
				os.mkdir(out_dir)
			dsagg = loompy.connect(self.input()[0].fn)
			logging.info("Computing auto-annotation")
			aa = cg.AutoAnnotator(root=cg.paths().autoannotation)
			aa.annotate_loom(dsagg)
			aa.save_in_loom(dsagg)

			dsagg.export(os.path.join(out_dir, "L1_" + self.tissue + "_expression.tab"))
			dsagg.export(os.path.join(out_dir, "L1_" + self.tissue + "_enrichment.tab"), layer="enrichment")
			dsagg.export(os.path.join(out_dir, "L1_" + self.tissue + "_enrichment_q.tab"), layer="enrichment_q")
			dsagg.export(os.path.join(out_dir, "L1_" + self.tissue + "_trinaries.tab"), layer="trinaries")

			ds = loompy.connect(self.input()[1].fn)

			logging.info("Plotting manifold graph with auto-annotation")
			tags = list(dsagg.col_attrs["AutoAnnotation"])
			cg.plot_graph(ds, os.path.join(out_dir, "L1_" + self.tissue + "_manifold.aa.png"), tags)

			logging.info("Plotting manifold graph with auto-auto-annotation")
			tags = list(dsagg.col_attrs["MarkerGenes"])
			cg.plot_graph(ds, os.path.join(out_dir, "L1_" + self.tissue + "_manifold.aaa.png"), tags)

			logging.info("Plotting marker heatmap")
			cg.plot_markerheatmap(ds, dsagg, n_markers_per_cluster=self.n_markers, out_file=os.path.join(out_dir, "L1_" + self.tissue + "_heatmap.pdf"))
