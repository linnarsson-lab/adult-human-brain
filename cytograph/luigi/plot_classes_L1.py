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


class PlotClassesL1(luigi.Task):
	"""
	Luigi Task to plot the MKNN graph
	"""
	tissue = luigi.Parameter()

	def requires(self) -> List[luigi.Task]:
		return cg.ClusterLayoutL1(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.tissue + ".classes.png"))

	def run(self) -> None:
		logging.info("Plotting classification of MKNN graph")
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_classes(ds, out_file)