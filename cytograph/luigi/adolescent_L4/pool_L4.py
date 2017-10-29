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
		targets = [
			"SpinalCord_Inhibitory",
			"SpinalCord_Excitatory",
			"Peripheral_Neurons",
			"Hypothalamus_Peptidergic",
			"Hindbrain_Inhibitory",
			"Hindbrain_Excitatory",
			"Brain_Neuroblasts",
			"Forebrain_Inhibitory",
			"Forebrain_Excitatory",
			"DiMesencephalon_Inhibitory",
			"DiMesencephalon_Excitatory",
			"Brain_Granule",
			"Brain_CholinergicMonoaminergic",
			"Striatum_MSN"
		]

		classes = ["Oligos", "Astrocytes", "Ependymal", "Vascular", "Immune", "PeripheralGlia"]
		for target in targets:
			yield cg.ClusterL3(target=target)
		for cl in classes:
			yield cg.FilterL2(major_class=cl, tissue="All")

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L4_All.loom"))
		
	def run(self) -> None:
		logging = cg.logging(self)
		samples = [x.fn for x in self.input()]
		max_cluster_id = 0
		cluster_ids: List[int] = []
		original_ids: List[int] = []
		samples_per_cell: List[int] = []
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
					original_ids += list(ca["Clusters"])
					samples_per_cell += [sample] * selection.shape[0]
					if dsout is None:
						dsout = loompy.create(out_file, vals[ordering, :], ds.row_attrs, ca)
					else:
						dsout.add_columns(vals[ordering, :], ca)
				max_cluster_id = max(cluster_ids) + 1
				ds.close()
			dsout.set_attr("Clusters", np.array(cluster_ids), axis=1)
			dsout.set_attr("OriginalClusters", np.array(original_ids), axis=1)
			dsout.set_attr("MajorTarget", np.array(samples_per_cell), axis=1)
			dsout.close()
