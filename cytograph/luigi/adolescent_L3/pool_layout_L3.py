from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi


class PoolLayoutL3(luigi.Task):
	"""
	Subsample every cluster by taking 20 cells per cluster and pooling into a single output file, then layout
	"""
	project = luigi.Parameter(default="Adolescent")
	n_cells = luigi.IntParameter(default=20)

	def requires(self) -> Iterator[luigi.Task]:
		if self.project == "Adolescent":
			tissues = cg.PoolSpec().tissues_for_project(self.project)
			classes = ["Oligos", "Astrocyte", "Cycling", "Vascular", "Immune", "Erythrocyte"]
			for tissue in tissues:
				yield cg.ClusterLayoutL2(project=self.project, tissue=tissue, major_class="Neurons")
			for cls in classes:
				yield cg.ClusterLayoutL2(project=self.project, tissue="All", major_class=cls)
		else:
			yield cg.ClusterLayoutL2(project=self.project, tissue="All", major_class="Development")

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths.build(), self.project + ".L3.loom"))
		
	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			max_cluster = 0
			for clustered in self.input():
				ds = loompy.connect(clustered.fn)
				# select cells
				labels = ds.col_attrs["Clusters"]
				max_cluster = max_cluster + np.max(labels)
				temp = []  # type: List[int]
				for i in range(max(labels) + 1):
					temp += list(np.random.choice(np.where(labels == i)[0], size=self.n_cells))
				cells = np.array(temp)
				# put the cells in the training and validation datasets
				for (ix, selection, vals) in ds.batch_scan(cells=cells, axis=1):
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][selection]
					ca["Clusters"] = ca["Clusters"] + max_cluster
					if dsout is None:
						loompy.create(self.output().fn, vals, row_attrs=ds.row_attrs, col_attrs=ca)
						dsout = loompy.connect(self.output().fn)
					else:
						dsout.add_columns(vals, ca)
			
			
			# BROKEN
			
			cg.cluster_layout(dsout, keep_existing_clusters=True)
			dsout.close()
