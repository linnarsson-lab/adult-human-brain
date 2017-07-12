from typing import *
import os
from math import exp, lgamma, log
import logging
import pandas as pd
import loompy
from scipy.special import beta, betainc, betaln
import numpy as np
import cytograph as cg
import luigi


class AutoAnnotateDev(luigi.Task):
	"""
	Luigi Task to auto-annotate clusters, level 2
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")
	time = luigi.Parameter(default="E7-E18")

	def requires(self) -> luigi.Task:
		return cg.TrinarizeDev(lineage=self.lineage, target=self.target, time=self.time)

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".aa.tab"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.aa.tab" % (self.lineage, self.target, self.time)))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			aa = cg.AutoAnnotator(root=cg.paths().autoannotation)
			aa.annotate(self.input().fn)
			aa.save(out_file)
