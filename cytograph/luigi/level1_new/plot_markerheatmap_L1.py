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


class PlotMarkerheatmapL1(luigi.Task):
	"""
	Luigi Task to plot the marker heatmap, level 1
	"""
	tissue = luigi.Parameter()
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [
			cg.PrepareTissuePool(tissue=self.tissue),
			cg.AggregateL1(tissue=self.tissue)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "L1_" + self.tissue + ".heatmap.pdf"))

	def run(self) -> None:
		logging.info("Plotting marker heatmap")
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			dsagg = loompy.connect(self.input()[1].fn)
			cg.plot_markerheatmap(ds, dsagg, n_markers_per_cluster=self.n_markers, out_file=out_file)
