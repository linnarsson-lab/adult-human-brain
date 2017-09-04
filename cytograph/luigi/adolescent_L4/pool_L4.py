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
import scipy.stats


class PoolL4(luigi.Task):
	"""
	Level 4 pooling of the adolescent dataset
	"""

	def requires(self) -> luigi.Task:
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		classes = ["Oligos", "AstroEpendymal", "Vascular", "Immune", "PeripheralGlia"]
		for tissue in tissues:
			yield cg.ClusterL3(tissue=tissue, major_class="Neurons")
		for cls in classes:
			yield cg.ClusterL3(tissue="All", major_class=cls)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L4_All.loom"))
		
	def run(self) -> None:
		samples = [x.fn for x in self.input()]
		max_cluster_id = 0
		cluster_ids: List[int] = []
		with self.output().temporary_path() as out_file:
			dsout: loompy.LoomConnection = None
			accessions = None  # type: np.ndarray
			for sample in samples:
				ds = loompy.connect(sample)
				if accessions is None:
					accessions = ds.row_attrs["Accession"]
				logging.info(f"Adding {ds.shape[1]} cells from {sample}")
				ordering = np.where(ds.row_attrs["Accession"][None, :] == accessions[:, None])[1]
				for (ix, selection, vals) in ds.batch_scan(axis=1, batch_size=cg.memory().axis1):
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][selection]
					cluster_ids += list(ca["Clusters"] + max_cluster_id)
					if dsout is None:
						dsout = loompy.create(out_file, vals[ordering, :], ds.row_attrs, ca)
					else:
						dsout.add_columns(vals[ordering, :], ca)
				max_cluster_id = max(cluster_ids) + 1
				ds.close()
			dsout.set_attr("Clusters", np.array(cluster_ids), axis=1)
			dsout.close()
