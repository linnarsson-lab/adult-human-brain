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
import scipy.cluster.hierarchy as hc


class AggregateAnalysis(luigi.Task):  # Status: Ok
	"""
	Aggregate all clusters in a new Loom file
	"""
	analysis = luigi.Parameter()
	n_markers = luigi.IntParameter(default=10)
	n_auto_genes = luigi.IntParameter(default=6)

	def requires(self) -> List[luigi.Task]:
		return cg.ClusterAnalysis(analysis=self.analysis)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "Analysis_" + self.analysis + ".agg.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.Aggregator(self.n_markers).aggregate(ds, out_file)
			dsagg = loompy.connect(out_file)

			logging.info("Computing auto-annotation")
			aa = cg.AutoAnnotator(root=cg.paths().autoannotation)
			aa.annotate_loom(dsagg)
			aa.save_in_loom(dsagg)
			ds.close()

			logging.info("Computing auto-auto-annotation")
			n_clusters = dsagg.shape[1]
			(selected, selectivity, specificity, robustness) = cg.AutoAutoAnnotator(n_genes=self.n_auto_genes).fit(dsagg)
			dsagg.set_attr("MarkerGenes", np.array([" ".join(ds.Gene[selected[:, ix]]) for ix in np.arange(n_clusters)]), axis=1)
			np.set_printoptions(precision=1, suppress=True)
			dsagg.set_attr("MarkerSelectivity", np.array([str(selectivity[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
			dsagg.set_attr("MarkerSpecificity", np.array([str(specificity[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
			dsagg.set_attr("MarkerRobustness", np.array([str(robustness[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
			dsagg.close()
