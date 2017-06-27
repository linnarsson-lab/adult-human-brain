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


class AggregateL1(luigi.Task):
	"""
	Aggregate all clusters in a new Loom file
	"""
	tissue = luigi.Parameter()
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [
			cg.PrepareTissuePool(tissue=self.tissue),
			cg.ClusterL1(tissue=self.tissue)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L1_" + self.tissue + ".agg.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			cg.Aggregator(self.n_markers).aggregate(ds, out_file, batch_size=cg.memory().axis0)

