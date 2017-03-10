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


class PlotGraphLineage(luigi.Task):
	"""
	Luigi Task to plot the MKNN graph, level 2
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")

	def requires(self) -> List[luigi.Task]:
		return [cg.ClusterLayoutLineage(lineage=self.lineage, target=self.target), cg.AutoAnnotateLineage(lineage=self.lineage, target=self.target)]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.lineage + "_" + self.target + ".mknn.png"))

	def run(self) -> None:
		logging.info("Plotting MKNN graph")
		# Parse the auto-annotation tags
		tags = []
		with open(self.input()[1].fn, "r") as f:
			content = f.readlines()[1:]
			for line in content:
				tags.append(line.split('\t')[1].replace(",", "\n")[:-1])
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			cg.plot_graph(ds, out_file, tags)
