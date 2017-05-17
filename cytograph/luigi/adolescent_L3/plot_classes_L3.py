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


class PlotClassesL3(luigi.Task):
	"""
	Luigi Task to plot the classification at L3
	"""
	project = luigi.Parameter(default="Adolescent")

	def requires(self) -> List[luigi.Task]:
		return cg.PoolLayoutL3(project=self.project)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.project + ".classes.png"))

	def run(self) -> None:
		logging.info("Plotting classification of MKNN graph")
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_classification(ds, out_file)
