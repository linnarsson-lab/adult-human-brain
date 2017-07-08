from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
import numpy_groupies.aggregate_numpy as npg
from statistics import mode


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
			return [(cg.PrepareTissuePool(tissue=tissue), cg.ClusterL1(tissue=tissue)) for tissue in tissues]
		else:
			return [(cg.PrepareTissuePool(tissue=self.tissue), cg.ClusterL1(tissue=self.tissue))]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L2_" + self.major_class + "_" + self.tissue + ".loom"))
		
	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			for (prepared, clustered) in self.input():
				ds = loompy.connect(prepared.fn)
				logging.info("Split/pool from " + prepared.fn)
				labels = ds.col_attrs["Class"]
				# Mask out cells that do not have the majority label of its cluster
				clusters = ds.col_attrs["Clusters"]
				majority_labels = npg.aggregate(clusters, labels, func=mode).astype('str')

				cells = []
				for ix in range(ds.shape[1]):
					if labels[ix] == self.major_class and labels[ix] == majority_labels[clusters[ix]]:
						cells.append(ix)
				logging.info(labels)
				for (ix, selection, vals) in ds.batch_scan(cells=np.array(cells), axis=1, batch_size=cg.memory().axis1):
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][selection]
					if dsout is None:
						dsout = loompy.create(out_file, vals, ds.row_attrs, ca)
					else:
						dsout.add_columns(vals, ca)
