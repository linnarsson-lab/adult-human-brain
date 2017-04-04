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

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutLineage(lineage=self.lineage, target=self.target)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.lineage + "_" + self.target + ".clusteravg.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			avgr = cg.Averager()
			ds = loompy.connect(self.input().fn)
			avgr.calculate_and_save(ds, out_file)
