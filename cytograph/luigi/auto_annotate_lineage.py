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


class AutoAnnotateLineage(luigi.Task):
	"""
	Luigi Task to auto-annotate clusters, level 2
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.TrinarizeLineage(lineage=self.lineage, target=self.target)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.lineage + "_" + self.target + ".aa.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			aa = cg.AutoAnnotator()
			aa.annotate(self.input().fn)
			aa.save(out_file)
