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


class AutoAnnotateL1(luigi.Task):
	"""
	Luigi Task to auto-annotate clusters, level 2
	"""
	tissue = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.AggregateL1(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L1_" + self.tissue + ".aa.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			aa = cg.AutoAnnotator()
			aa.annotate_loom(ds)
			aa.save(out_file)
			aa.save_in_loom(ds)
			ds.close()
