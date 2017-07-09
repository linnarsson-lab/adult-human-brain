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


class autoannotate(luigi.Config):
	species = luigi.Parameter(default="Mm")


class AggregateL1(luigi.Task):
	"""
	Aggregate all clusters in a new Loom file
	"""
	tissue = luigi.Parameter()
	n_markers = luigi.IntParameter(default=10)
	n_auto_genes = luigi.IntParameter(default=6)

	def requires(self) -> List[luigi.Task]:
		return cg.ClusterL1(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L1_" + self.tissue + ".agg.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			cg.Aggregator(self.n_markers).aggregate(ds, out_file)
			dsagg = loompy.connect(out_file)

			logging.info("Computing auto-annotation")
			if autoannotate().species == "Mm":
				aa = cg.AutoAnnotator()
			elif autoannotate().species == "Hs":
				aa = cg.AutoAnnotator(root="../auto-annotationHs")
			else:
				raise ValueError("%s is not a valid autoannotate-species, try with Mm/Hs" % autoannotate().species)
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