from typing import *
import os
import logging
from math import exp, lgamma, log
import loompy
from scipy.special import beta, betainc, betaln
import numpy as np
import cytograph as cg
import luigi


class ExpressionAverageLineage(luigi.Task):
	"""
	Luigi Task to calculate the average expression for cluster
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")
	time = luigi.Parameter(default="E7-E18")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutDev(lineage=self.lineage, target=self.target, time=self.time)

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".clusteravg.loom"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.clusteravg.loom" % (self.lineage, self.target, self.time)))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			avgr = cg.Averager()
			ds = loompy.connect(self.input().fn)
			avgr.calculate_and_save(ds, out_file)
