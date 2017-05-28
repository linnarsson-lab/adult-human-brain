from typing import *
import os
import logging
from math import exp, lgamma, log
import loompy
from scipy.special import beta, betainc, betaln
import numpy as np
import cytograph as cg
import luigi


class ExpressionAverageProcess(luigi.Task):
	"""
	Luigi Task to calculate the average expression for cluster
	"""
	processname = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutProcess(processname=self.processname)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "%s.clusteravg.loom" % self.processname))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			avgr = cg.Averager()
			ds = loompy.connect(self.input().fn)
			avgr.calculate_and_save(ds, out_file)
