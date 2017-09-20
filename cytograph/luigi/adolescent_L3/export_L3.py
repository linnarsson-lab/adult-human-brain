from typing import *
import os
#import logging
import loompy
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import numpy as np
import networkx as nx
import cytograph as cg
import luigi


class ExportL3(luigi.Task):
	"""
	Luigi Task to export summary files
	"""
	target = luigi.Parameter()  # e.g. Forebrain_Excitatory
	n_markers = luigi.IntParameter(default=10)

	def requires(self) -> List[luigi.Task]:
		return [
			cg.AggregateL3(target=self.target),
			cg.ClusterL3(target=self.target)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L3_" + self.target + "_exported"))

	def run(self) -> None:
		logging = cg.logging(self)
		with self.output().temporary_path() as out_dir:
			logging.info("Exporting cluster data")
			if not os.path.exists(out_dir):
				os.mkdir(out_dir)
			dsagg = loompy.connect(self.input()[0].fn)
			logging.info("Computing auto-annotation")
			aa = cg.AutoAnnotator(root=cg.paths().autoannotation)
			aa.annotate_loom(dsagg)
			aa.save_in_loom(dsagg)

			dsagg.export(os.path.join(out_dir, "L3_" + self.target + "_expression.tab"))
			dsagg.export(os.path.join(out_dir, "L3_" + self.target + "_enrichment.tab"), layer="enrichment")
			dsagg.export(os.path.join(out_dir, "L3_" + self.target + "_enrichment_q.tab"), layer="enrichment_q")
			dsagg.export(os.path.join(out_dir, "L3_" + self.target + "_trinaries.tab"), layer="trinaries")

			logging.info("Plotting manifold graph with auto-annotation")
			tags = list(dsagg.col_attrs["AutoAnnotation"][np.argsort(dsagg.col_attrs["Clusters"])])
			ds = loompy.connect(self.input()[1].fn)
			cg.plot_graph(ds, os.path.join(out_dir, "L3_" + self.target + "_manifold.aa.png"), tags)

			logging.info("Plotting manifold graph with auto-auto-annotation")
			tags = list(dsagg.col_attrs["MarkerGenes"][np.argsort(dsagg.col_attrs["Clusters"])])
			cg.plot_graph(ds, os.path.join(out_dir, "L3_" + self.target + "_manifold.aaa.png"), tags)

			logging.info("Plotting marker heatmap")
			cg.plot_markerheatmap(ds, dsagg, n_markers_per_cluster=self.n_markers, out_file=os.path.join(out_dir, "L3_" + self.target + "_heatmap.pdf"))

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
			with open(os.path.join(out_dir, "L3_" + self.target + "_distances.txt"), "w") as f:
				f.write(str(np.diag(D, k=1)))
