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


class AutoAnnotateProcess(luigi.Task):
	"""
	Luigi Task to auto-annotate clusters, level 2
	"""
	processname = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.TrinarizeProcess(processname=self.processname)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "%s.aa.tab" % self.processname))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			aa = cg.AutoAnnotator(root=cg.paths().autoannotation)
			aa.annotate(self.input().fn)
			aa.save(out_file)
