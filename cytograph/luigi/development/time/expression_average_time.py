from typing import *
import os
import logging
from math import exp, lgamma, log
import loompy
from scipy.special import beta, betainc, betaln
import numpy as np
import cytograph as cg
import luigi


class ExpressionAverageTime(luigi.Task):
	"""
	Luigi Task to calculate the average expression for cluster
	"""
	time = luigi.Parameter(default="E7-E18")
	lineage = luigi.Parameter(default="All")
	target = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutDev(lineage=self.lineage, target=self.target, time=self.time)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.timeavg.loom" % (self.lineage, self.target, self.time)))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			avgr = cg.Averager(func="mean")
			ds = loompy.connect(self.input().fn)
			avgr.calculate_and_save(ds=ds, output_file=out_file, aggregator_class="Age", category_summary=("Clusters", "SampleID"))
