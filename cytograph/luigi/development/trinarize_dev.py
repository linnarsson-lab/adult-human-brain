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


class TrinarizeDev(luigi.Task):
	"""
	Luigi Task to calculate trinarization of genes across clusters
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")
	time = luigi.Parameter(default="E7-E18")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutDev(lineage=self.lineage, target=self.target, time=self.time)

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".trinary.tab"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.trinary.tab" % (self.lineage, self.target, self.time)))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			tr = cg.Trinarizer(trinarization().f)
			tr.fit(ds)
			tr.save(out_file)
