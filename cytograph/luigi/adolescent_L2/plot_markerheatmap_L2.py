from typing import *
import os
import logging
import loompy
import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi
from palettable.tableau import Tableau_20


class PlotMarkerheatmapL2(luigi.Task):
	"""
	Luigi Task to plot the marker heatmap, level 2
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [
			cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class),
			cg.ClusterL2(tissue=self.tissue, major_class=self.major_class),
			cg.AggregateL2(tissue=self.tissue, major_class=self.major_class)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "L2_" + self.major_class + "_" + self.tissue + ".heatmap.pdf"))

	def run(self) -> None:
		logging.info("Plotting marker heatmap")
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			dsagg = loompy.connect(self.input()[2].fn)
			cg.plot_markerheatmap(ds, dsagg, n_markers_per_cluster=self.n_markers, out_file=out_file)
