from typing import *
import os
import csv
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi
import scipy.cluster.hierarchy as hierarchy
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc


class AggregateL3(luigi.Task):
	"""
	Aggregate all clusters in a new Loom file
	"""

	def requires(self) -> List[luigi.Task]:
		return cg.AggregateL2(tissue=self.tissue, major_class=self.major_class)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "L3_All.agg.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.Aggregator(self.n_markers).aggregate(ds, out_file)