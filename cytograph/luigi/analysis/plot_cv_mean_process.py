from typing import *
import os
import loompy
import numpy as np
import cytograph as cg
import luigi


class PlotCVMeanProcess(luigi.Task):
	"""
	Luigi Task to plot CV vs mean, indicating selected genes for level 2 clustering
	"""
	processname = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutProcess(processname=self.processname)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "%s.CV_mean.png" % self.processname))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_cv_mean(ds, out_file)
