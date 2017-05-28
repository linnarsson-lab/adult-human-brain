from typing import *
import os
import csv
import logging
from shutil import copyfile
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
import scipy.cluster.hierarchy as hierarchy
import numpy_groupies.aggregate_numpy as npg


def _renumber_clusters(ds: loompy.LoomConnection) -> np.ndarray:
	offset = 0
	clusters = np.zeros(ds.shape[1])
	for tissue in set(ds.col_attrs["TissuePool"]):
		indices = ds.col_attrs["TissuePool"] == tissue
		labels = ds.col_attrs["Clusters"][indices]
		start = min(labels)
		labels = labels - start + offset
		if len(set(labels)) != max(labels) - min(labels) + 1:
			raise ValueError("Cannot renumber when some cluster labels are unused")
		clusters[indices] = labels
		offset = max(labels)
	return clusters.astype('int')


class PoolAllL3(luigi.Task):
	"""
	Aggregate all clusters in a new Loom file
	"""

	def requires(self) -> Iterator[luigi.Task]:
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		classes = ["Oligos", "Astrocyte", "Cycling", "Vascular", "Immune", "Erythrocyte"]
		for tissue in tissues:
			yield cg.ClusterLayoutL2(project="Adolescent", tissue=tissue, major_class="Neurons")
		for cls in classes:
			yield cg.ClusterLayoutL2(project="Adolescent", tissue="All", major_class=cls)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths.build, "Adolescent.L3.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			index = 0
			for cl in self.input():
				logging.info("Appending: " + cl.fn)
				ds = loompy.connect(cl.fn)
				tissue = cl.fn.split("/")[1].split(".")[0]
				ds.set_attr("TissuePool", np.array([tissue] * ds.shape[1]), axis=1)
				if dsout is None:
					copyfile(cl.fn, out_file)
					dsout = loompy.connect(out_file)
				else:
					dsout.add_loom(cl.fn)
			
			dsout.set_attr("Clusters", _renumber_clusters(dsout), axis=1)
			dsout.close()
