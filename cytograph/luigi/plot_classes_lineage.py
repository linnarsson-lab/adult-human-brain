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


class PlotClassesLineage(luigi.Task):
	"""
	Luigi Task to plot the MKNN graph

	TODO: Make 3d tsne
	"""
	lineage = luigi.Parameter(default="Ectodermal")  # Alternativelly Endomesodermal
	target = luigi.Parameter(default="All")  # one between Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral,\ForebrainVentrothalamic, Midbrain, Hindbrain

	def requires(self) -> List[luigi.Task]:
		return cg.ClusterLayoutLineage(lineage=self.lineage, target=self.target)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.lineage + "_" + self.target + ".classes.png"))

	def run(self) -> None:
		logging.info("Plotting classification of MKNN graph")
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_classes(ds, out_file)