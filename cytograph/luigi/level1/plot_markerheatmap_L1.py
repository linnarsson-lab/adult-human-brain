from typing import *
import os
import logging
import loompy
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi


class PlotMarkerheatmapL2(luigi.Task):
	"""
	Luigi Task to plot the marker heatmap, level 2
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> List[luigi.Task]:
		return cg.ClusterLayoutL2(tissue=self.tissue, major_class=self.major_class, project=self.project)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths.build(), self.major_class + "_" + self.tissue + ".heatmap.pdf"))

	def run(self) -> None:
		logging.info("Plotting marker heatmap")
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_markerheatmap(ds, out_file)
