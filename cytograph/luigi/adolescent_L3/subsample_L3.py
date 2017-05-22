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


class SubsampleL3(luigi.Task):
	"""
	Aggregate all clusters in a new Loom file
	"""

	def requires(self) -> Iterator[luigi.Task]:
		return cg.PoolAllL3()

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "Adolescent.L3.subsample.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			ds = loompy.connect(self.input().fn)
			for ix in range(max(ds.col_attrs["Clusters"]) + 1):
				cells = np.where(ds.col_attrs["Clusters"] == ix)[0]
				n_chosen = min(50, len(cells))
				logging.info(ds.col_attrs["TissuePool"][cells[0]] + " cluster #" + str(ix) + " (" + str(n_chosen) + " of " + str(len(cells)) + ")")
				subset = sorted(np.random.choice(cells, n_chosen, replace=False))
				m = ds[:, subset]
				ca = {key: v[subset] for key, v in ds.col_attrs.items()}
				if dsout is None:
					dsout = loompy.create(out_file, m, ds.row_attrs, ca)
				else:
					dsout.add_columns(m, ca)
