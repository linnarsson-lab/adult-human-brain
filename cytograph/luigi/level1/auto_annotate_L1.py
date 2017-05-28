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


class autoannotate(luigi.Config):
	species = luigi.Parameter(default="Mm")


class AutoAnnotateL1(luigi.Task):
	"""
	Luigi Task to auto-annotate clusters
	"""
	tissue = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.TrinarizeL1(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths.build(), self.tissue + ".aa.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			if autoannotate().species == "Mm":
				aa = cg.AutoAnnotator()
			elif autoannotate().species == "Hs":
				aa = cg.AutoAnnotator(root="../auto-annotationHs")
			else:
				raise ValueError("%s is not a valid autoannotate-species, try with Mm/Hs" % autoannotate().species)
			aa.annotate(self.input().fn)
			aa.save(out_file)
