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


class TrinarizeL2(luigi.Task):
	"""
	Luigi Task to calculate trinarization of genes across clusters
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return [
			cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class, project=self.project),
			cg.ClusterL2(tissue=self.tissue, major_class=self.major_class, project=self.project)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".trinary.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			tr = cg.Trinarizer(trinarization().f)
			tr.fit(ds)
			tr.save(out_file)
