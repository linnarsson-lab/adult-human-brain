from typing import *
import os
import logging
import loompy
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi


class PlotClassesLineage(luigi.Task):
	"""
	Luigi Task to plot the MKNN graph

	TODO: Make 3d tsne
	"""
	lineage = luigi.Parameter(default="Ectodermal")  # Alternativelly Endomesodermal
	target = luigi.Parameter(default="All")  # one between Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral,\ForebrainVentrothalamic, Midbrain, Hindbrain
	time = luigi.Parameter(default="E7-E18") 

	def requires(self) -> List[luigi.Task]:
		return cg.ClusterLayoutDev(lineage=self.lineage, target=self.target, time=self.time)

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".classes.png"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.classes.png" % (self.lineage, self.target, self.time)))

	def run(self) -> None:
		logging.info("Plotting classification of MKNN graph")
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_classes(ds, out_file)
