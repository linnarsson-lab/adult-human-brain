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


class PlotClassesL2(luigi.Task):
	"""
	Luigi Task to plot the MKNN graph
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> List[luigi.Task]:
		return [
			cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class, project=self.project),
			cg.ClusterL2(tissue=self.tissue, major_class=self.major_class, project=self.project)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".classes.png"))

	def run(self) -> None:
		logging.info("Plotting classification of MKNN graph")
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			cg.plot_classes(ds, out_file)
