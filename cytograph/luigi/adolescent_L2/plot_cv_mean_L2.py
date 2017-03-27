from typing import *
import os
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class PlotCVMeanL2(luigi.Task):
	"""
	Luigi Task to plot CV vs mean, indicating selected genes for level 2 clustering
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutL2(tissue=self.tissue, major_class=self.major_class, project=self.project)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".CV_mean.png"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_cv_mean(ds, out_file)
