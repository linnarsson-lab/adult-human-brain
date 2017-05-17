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


class PlotManifoldL2(luigi.Task):
	"""
	Luigi Task to plot the multiscale KNN graph, level 2
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> List[luigi.Task]:
		return [
			cg.AutoAnnotateL2(tissue=self.tissue, major_class=self.major_class),
			cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "L2_" + self.major_class + "_" + self.tissue + ".manifold.png"))

	def run(self) -> None:
		logging.info("Plotting MKNN graph")
		# Parse the auto-annotation tags
		tags = []
		with open(self.input()[0].fn, "r") as f:
			content = f.readlines()[1:]
			for line in content:
				tags.append(line.split('\t')[1].replace(",", "\n")[:-1])
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[1].fn)
			cg.plot_graph(ds, out_file, tags)
