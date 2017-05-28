from typing import *
import os
import logging
import loompy
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi


class PlotGraphDev(luigi.Task):
	"""
	Luigi Task to plot the MKNN graph, level 2
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")
	time = luigi.Parameter(default="E7-E18")

	def requires(self) -> List[luigi.Task]:
		return [cg.ClusterLayoutDev(lineage=self.lineage, target=self.target, time=self.time), cg.AutoAnnotateDev(lineage=self.lineage, target=self.target, time=self.time)]

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".mknn.png"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.mknn.png" % (self.lineage, self.target, self.time)))

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
