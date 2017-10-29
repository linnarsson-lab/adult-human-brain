from typing import *
import os
import csv
#import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
import numpy_groupies.aggregate_numpy as npg
import scipy.stats
import tempfile


# params = {  # eps_pct and min_pts
# 	"L2_Neurons_Amygdala": [60, 40],
# 	"L2_Neurons_Cerebellum": [80, 20],
# 	"L2_Neurons_Cortex1": [70, 20],
# 	"L2_Neurons_Cortex2": [50, 40],
# 	"L2_Neurons_Cortex3": [60, 20],
# 	"L2_Neurons_DRG": [75, 10],
# 	"L2_Neurons_Enteric": [60, 10],
# 	"L2_Neurons_Hippocampus": [90, 10],
# 	"L2_Neurons_Hypothalamus": [75, 20],
# 	"L2_Neurons_Medulla": [60, 20],
# 	"L2_Neurons_MidbrainDorsal": [60, 20],
# 	"L2_Neurons_MidbrainVentral": [60, 20],
# 	"L2_Neurons_Olfactory": [70, 40],
# 	"L2_Neurons_Pons": [60, 20],
# 	"L2_Neurons_SpinalCord": [90, 10],
# 	"L2_Neurons_StriatumDorsal": [80, 40],
# 	"L2_Neurons_StriatumVentral": [80, 20],
# 	"L2_Neurons_Sympathetic": [60, 10],
# 	"L2_Neurons_Thalamus": [75, 20],
# 	"L2_Oligos_All": [95, 500],
# 	"L2_AstroEpendymal_All": [80, 40],
# 	"L2_Blood_All": [70, 20],
# 	"L2_Immune_All": [70, 70],
# 	"L2_PeripheralGlia_All": [80, 20],
# 	"L2_Vascular_All": [80, 100]
# }

# params = {  # eps_pct and min_pts
# 	"L2_Neurons_Amygdala": [85, 10],
# 	"L2_Neurons_Cerebellum": [90, 10],
# 	"L2_Neurons_Cortex1": [80, 10],
# 	"L2_Neurons_Cortex2": [75, 10],
# 	"L2_Neurons_Cortex3": [85, 10],
# 	"L2_Neurons_DRG": [70, 10],
# 	"L2_Neurons_Enteric": [60, 10],
# 	"L2_Neurons_Hippocampus": [90, 10],
# 	"L2_Neurons_Hypothalamus": [80, 10],
# 	"L2_Neurons_Medulla": [70, 10],
# 	"L2_Neurons_MidbrainDorsal": [90, 10],
# 	"L2_Neurons_MidbrainVentral": [80, 10],
# 	"L2_Neurons_Olfactory": [80, 10],
# 	"L2_Neurons_Pons": [75, 10],
# 	"L2_Neurons_SpinalCord": [90, 10],
# 	"L2_Neurons_StriatumDorsal": [90, 10],
# 	"L2_Neurons_StriatumVentral": [85, 10],
# 	"L2_Neurons_Sympathetic": [60, 10],
# 	"L2_Neurons_Thalamus": [90, 10],
# 	"L2_Oligos_All": [95, 500],
# 	"L2_AstroEpendymal_All": [80, 40],
# 	"L2_Blood_All": [70, 20],
# 	"L2_Immune_All": [70, 70],
# 	"L2_PeripheralGlia_All": [80, 20],
# 	"L2_Vascular_All": [90, 10]
# }

params = {  # eps_pct and min_pts
    "L2_Neurons_Amygdala": [80, 10],
    "L2_Neurons_Cerebellum": [90, 10],
    "L2_Neurons_Cortex1": [75, 10],
    "L2_Neurons_Cortex2": [75, 10],
    "L2_Neurons_Cortex3": [80, 10],
    "L2_Neurons_DRG": [70, 10],
    "L2_Neurons_Enteric": [60, 10],
    "L2_Neurons_Hippocampus": [90, 10],
    "L2_Neurons_Hypothalamus": [75, 10],
    "L2_Neurons_Medulla": [70, 10],
    "L2_Neurons_MidbrainDorsal": [80, 10],
    "L2_Neurons_MidbrainVentral": [75, 10],
    "L2_Neurons_Olfactory": [75, 10],
    "L2_Neurons_Pons": [75, 10],
    "L2_Neurons_SpinalCord": [90, 10],
    "L2_Neurons_StriatumDorsal": [80, 10],
    "L2_Neurons_StriatumVentral": [75, 10],
    "L2_Neurons_Sympathetic": [60, 10],
    "L2_Neurons_Thalamus": [75, 10],
    "L2_Oligos_All": [95, 500],
    "L2_Astrocytes_All": [70, 40],
    "L2_Ependymal_All": [70, 40],
    "L2_Blood_All": [70, 20],
    "L2_Immune_All": [80, 40],
    "L2_PeripheralGlia_All": [80, 20],
    "L2_Vascular_All": [90, 10]
}


class ClusterL2(luigi.Task):
	"""
	Level 2 clustering of the adolescent dataset
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	n_genes = luigi.IntParameter(default=1000)
	n_components = luigi.IntParameter(default=30)
	k = luigi.IntParameter(default=5)
	N = luigi.IntParameter(default=5000)
	gtsne = luigi.BoolParameter(default=True)
	alpha = luigi.FloatParameter(default=1)

	def requires(self) -> luigi.Task:
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		if self.tissue == "All":
			return [cg.ClusterL1(tissue=tissue) for tissue in tissues]
		else:
			return [cg.ClusterL1(tissue=self.tissue)]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L2_" + self.major_class + "_" + self.tissue + ".loom"))
		
	def run(self) -> None:
		logging = cg.logging(self)
		dsout = None  # type: loompy.LoomConnection
		accessions = None  # type: np.ndarray
		with self.output().temporary_path() as out_file:
			for clustered in self.input():
				ds = loompy.connect(clustered.fn)
				logging.info("Split/pool from " + clustered.fn)
				labels = ds.col_attrs["Class"]

				# Mask out cells that do not have the majority label of its cluster
				clusters = ds.col_attrs["Clusters"]

				def mode(x):
					return scipy.stats.mode(x)[0][0]

				majority_labels = npg.aggregate(clusters, labels, func=mode).astype('str')

				cells = []
				for ix in range(ds.shape[1]):
					if labels[ix] == self.major_class and labels[ix] == majority_labels[clusters[ix]]:
						cells.append(ix)
				logging.info("Keeping " + str(len(cells)) + " cells with majority labels")
				if len(cells) == 0:
					continue

				# Keep track of the gene order in the first file
				if accessions is None:
					accessions = ds.row_attrs["Accession"]
				
				ordering = np.where(ds.row_attrs["Accession"][None, :] == accessions[:, None])[1]
				for (ix, selection, vals) in ds.batch_scan(cells=np.array(cells), axis=1, batch_size=cg.memory().axis1):
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][selection]
					if dsout is None:
						dsout = loompy.create(out_file, vals[ordering, :], ds.row_attrs, ca)
					else:
						dsout.add_columns(vals[ordering, :], ca)

			#
			# logging.info("Poisson imputation")
			# pi = cg.PoissonImputation(k=self.k, N=self.N, n_genes=self.n_genes, n_components=self.n_components)
			# pi.impute_inplace(dsout)

			logging.info("Learning the manifold")
			ds = loompy.connect(out_file)
			ml = cg.ManifoldLearning2(n_genes=self.n_genes, gtsne=self.gtsne, alpha=self.alpha)
			(knn, mknn, tsne) = ml.fit(ds)
			ds.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			ds.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			ds.set_attr("_X", tsne[:, 0], axis=1)
			ds.set_attr("_Y", tsne[:, 1], axis=1)

			logging.info("Clustering on the manifold")
			fname = "L2_" + self.major_class + "_" + self.tissue
			(eps_pct, min_pts) = params[fname]
			cls = cg.Clustering(method="mknn_louvain", eps_pct=eps_pct, min_pts=min_pts)
			labels = cls.fit_predict(ds)
			ds.set_attr("Clusters", labels, axis=1)
			logging.info(f"Found {labels.max() + 1} clusters")
			cg.Merger(min_distance=0.2).merge(ds)
			logging.info(f"Merged to {ds.col_attrs['Clusters'].max() + 1} clusters")
			ds.close()
		dsout.close()
