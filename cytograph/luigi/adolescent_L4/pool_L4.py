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
		classes = ["Oligos", "AstroEpendymal", "Vascular", "Immune", "Blood", "PeripheralGlia"]
		for tissue in tissues:
			yield cg.ClusterL3(tissue=tissue, major_class="Neurons")
		for cls in classes:
			yield cg.ClusterL3(tissue="All", major_class=cls)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L4_All.loom"))
		
	def run(self) -> None:
		samples = [x.fn for x in self.input()]
		with self.output().temporary_path() as out_file:
			dsout: loompy.LoomConnection = None
			accessions = None  # type: np.ndarray
			for sample in samples:
				ds = loompy.connect(sample)
				if accessions is None:
					accessions = ds.row_attrs["Accession"]
				logging.info(f"Adding {ds.shape[1]} cells from {sample}")
				for (ix, selection, view) in ds.scan(axis=1, key={"Accession": accessions}):
					if dsout is None:
						dsout = loompy.create(out_file, view[:, :], view.row_attrs, view.col_attrs)
					else:
						dsout.add_columns(view[:, :], view.col_attrs)
				ds.close()
			dsout.close()
