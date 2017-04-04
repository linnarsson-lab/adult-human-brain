from typing import *
import os
import logging
from math import exp, lgamma, log
import loompy
from scipy.special import beta, betainc, betaln
import numpy as np
import cytograph as cg
import luigi


class trinarization(luigi.Config):
	f = luigi.FloatParameter(default=0.2)


class TrinarizeLineage(luigi.Task):
	"""
	Luigi Task to calculate trinarization of genes across clusters
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutLineage(lineage=self.lineage, target=self.target)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.lineage + "_" + self.target + ".trinary.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			tr = cg.Trinarizer(trinarization().f)
			tr.fit(ds)
			tr.save(out_file)
