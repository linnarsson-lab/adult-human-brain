from typing import *
import os
import loompy
import numpy as np
import cytograph as cg
import luigi


class PlotCVMeanLineage(luigi.Task):
	"""
	Luigi Task to plot CV vs mean, indicating selected genes for level 2 clustering
	"""
	lineage = luigi.Parameter(default="Ectodermal")  # Alternativelly Endomesodermal
	target = luigi.Parameter(default="All")  # one between Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral, ForebrainVentrothalamic, Midbrain, Hindbrain
	time = luigi.Parameter(default="E7-E18") 

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutDev(lineage=self.lineage, target=self.target, time=self.time)

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".CV_mean.png"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.CV_mean.png" % (self.lineage, self.target, self.time)))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.plot_cv_mean(ds, out_file)
