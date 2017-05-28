from typing import *
import os
import loompy
import numpy as np
import cytograph as cg
import luigi


class PlotCVMeanL1(luigi.Task):
	"""
	Luigi Task to plot CV vs mean, indicating selected genes
	"""
	tissue = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutL1(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, self.tissue + ".LJ.CV_mean.png"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_cv_mean(ds, out_file)
