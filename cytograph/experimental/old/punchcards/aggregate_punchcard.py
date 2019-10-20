from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import development_mouse as dm
import luigi
import scipy.cluster.hierarchy as hierarchy
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc


class AggregatePunchcard(luigi.Task):  # Status: Ok
	"""
	Summary statistics of all clusters in a new Loom file
	"""
	card = luigi.Parameter(description="Name of the punchcard")
	n_markers = luigi.IntParameter(default=10, description="Number of markers considered by the Aggergator")
	n_auto_genes = luigi.IntParameter(default=6, description="Number of genes to use in the AutoAutoannotation")

	def requires(self) -> List[luigi.Task]:
		return dm.ClusterPunchcard(card=self.card)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(dm.paths().build, f"{self.card}.agg.loom"))

	def run(self) -> None:
		logging = cg.logging(self)
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.Aggregator().aggregate(ds, out_file) # cg.Aggregator(self.n_markers).aggregate(ds, out_file) caused error
			dsagg = loompy.connect(out_file)

			logging.info("Computing auto-annotation")
			aa = cg.AutoAnnotator(root=dm.paths().autoannotation)
			aa.annotate_loom(dsagg)
			aa.save_in_loom(dsagg)
			

			logging.info("Computing auto-auto-annotation")
			n_clusters = dsagg.shape[1]
			(selected, selectivity, specificity, robustness) = cg.AutoAutoAnnotator(n_genes=self.n_auto_genes).fit(dsagg)
			dsagg.set_attr("MarkerGenes", np.array([" ".join(ds.row_attrs["Gene"][selected[:, ix]]) for ix in np.arange(n_clusters)]), axis=1)
			ds.close()
			np.set_printoptions(precision=1, suppress=True)
			dsagg.set_attr("MarkerSelectivity", np.array([str(selectivity[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
			dsagg.set_attr("MarkerSpecificity", np.array([str(specificity[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
			dsagg.set_attr("MarkerRobustness", np.array([str(robustness[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
			dsagg.close()
