from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi


class SplitAndPool(luigi.Task):
	"""
	Luigi Task to split the results of level 1 analysis by major cell class, and pool each class separately

	If tissue is "All", all tissues will be pooled.
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		if self.tissue == "All":
			return [cg.PrepareTissuePool(tissue=tissue) for tissue in tissues]
		else:
			return [cg.PrepareTissuePool(tissue=self.tissue)]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths.build(), "L2_" + self.major_class + "_" + self.tissue + ".loom"))
		
	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			for clustered in self.input():
				ds = loompy.connect(clustered.fn)
				logging.info("Split/pool from " + clustered.fn)
				labels = ds.col_attrs["Class"]
				for (ix, selection, vals) in ds.batch_scan(axis=1):
					subset = np.intersect1d(np.where(labels == self.major_class)[0], selection)
					if subset.shape[0] == 0:
						continue
					m = vals[:, subset - ix]
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][subset]
					if dsout is None:
						dsout = loompy.create(out_file, m, ds.row_attrs, ca)
					else:
						dsout.add_columns(m, ca)
