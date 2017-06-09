from typing import *
import os
import csv
import logging
import pickle
import loompy
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
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return cg.PoolAllL3()

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L3_Adolescent.agg.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.Aggregator(self.n_markers).aggregate(ds, out_file)
