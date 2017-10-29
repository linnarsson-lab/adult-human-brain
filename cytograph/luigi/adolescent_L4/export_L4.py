from typing import *
import os
import logging
import loompy
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi
from scipy.spatial.distance import squareform, pdist


class ExportL4(luigi.Task):
	"""
	Luigi Task to export summary files
	"""
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [cg.AggregateL4(), cg.PoolL4()]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L4_All_exported"))

	def run(self) -> None:
		logging = cg.logging(self, True)
		with self.output().temporary_path() as out_dir:
			logging.info("Exporting cluster data")
			if not os.path.exists(out_dir):
				os.mkdir(out_dir)
			dsagg = loompy.connect(self.input()[0].fn)
			dsagg.export(os.path.join(out_dir, "L4_All_expression.tab"))
			dsagg.export(os.path.join(out_dir, "L4_All_enrichment.tab"), layer="enrichment")
			dsagg.export(os.path.join(out_dir, "L4_All_enrichment_q.tab"), layer="enrichment_q")
			dsagg.export(os.path.join(out_dir, "L4_All_trinaries.tab"), layer="trinaries")

			logging.info("Computing discordance distances")
			pep = 0.05
			n_labels = dsagg.shape[1]

			def discordance_distance(a: np.ndarray, b: np.ndarray) -> float:
				"""
				Number of genes that are discordant with given PEP, divided by number of clusters
				"""
				return np.sum((1 - a) * b + a * (1 - b) > 1 - pep) / n_labels

			data = dsagg.layer["trinaries"][:n_labels * 10, :].T
			D = squareform(pdist(data, discordance_distance))
			with open(os.path.join(out_dir, "L4_All_distances.txt"), "w") as f:
				f.write(str(np.diag(D, k=1)))
