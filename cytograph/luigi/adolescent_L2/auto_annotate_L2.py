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


class AutoAnnotateL2(luigi.Task):
	"""
	Luigi Task to auto-annotate clusters, level 2
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.AggregateL2(tissue=self.tissue, major_class=self.major_class)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".aa.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			aa = cg.AutoAnnotator()
			aa.annotate_loom(self.input().fn)
			aa.save(out_file)
