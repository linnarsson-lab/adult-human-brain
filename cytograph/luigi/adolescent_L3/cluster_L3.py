from typing import *
import os
from shutil import copyfile
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
import numpy_groupies.aggregate_numpy as npg
import scipy.stats


params = {	# eps_pct and min_pts
	"L3_Neurons_Amygdala": [75, 20],
	"L3_Neurons_Cerebellum": [80, 20],
	"L3_Neurons_Cortex1": [70, 20],
	"L3_Neurons_Cortex2": [50, 40],
	"L3_Neurons_Cortex3": [60, 20],
	"L3_Neurons_DRG": [75, 10],
	"L3_Neurons_Enteric": [60, 10],
	"L3_Neurons_Hippocampus": [90, 10],
	"L3_Neurons_Hypothalamus": [75, 20],
	"L3_Neurons_Medulla": [60, 20],
	"L3_Neurons_MidbrainDorsal": [60, 20],
	"L3_Neurons_MidbrainVentral": [60, 20],
	"L3_Neurons_Olfactory": [70, 40],
	"L3_Neurons_Pons": [60, 20],
	"L3_Neurons_SpinalCord": [90, 10],
	"L3_Neurons_StriatumDorsal": [80, 40],
	"L3_Neurons_StriatumVentral": [80, 20],
	"L3_Neurons_Sympathetic": [70, 10],
	"L3_Neurons_Thalamus": [75, 20],
	"L3_Oligos_All": [95, 500],
	"L3_AstroEpendymal_All": [80, 70],
	"L3_Blood_All": [70, 20],
	"L3_Immune_All": [70, 70],
	"L3_PeripheralGlia_All": [75, 40],
	"L3_Vascular_All": [80, 100]
}


class ClusterL3(luigi.Task):
	"""
	Level 3 clustering of the adolescent dataset
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	method = luigi.Parameter(default='dbscan')  # or 'hdbscan'
	n_genes = luigi.IntParameter(default=1000)
	gtsne = luigi.BoolParameter(default=True)
	alpha = luigi.FloatParameter(default=1)
	pep = luigi.FloatParameter(default=0.01)

	def requires(self) -> luigi.Task:
		return cg.FilterL2(tissue=self.tissue, major_class=self.major_class)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L3_" + self.major_class + "_" + self.tissue + ".loom"))
		
	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			logging.info("Learning the manifold")
			copyfile(self.input().fn, out_file)
			ds = loompy.connect(out_file)
			ml = cg.ManifoldLearning(self.n_genes, self.gtsne, self.alpha)
			(knn, mknn, tsne) = ml.fit(ds)
			ds.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			ds.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			ds.set_attr("_X", tsne[:, 0], axis=1)
			ds.set_attr("_Y", tsne[:, 1], axis=1)

			logging.info("Clustering on the manifold")
			fname = "L3_" + self.major_class + "_" + self.tissue
			(eps_pct, min_pts) = params[fname]
			cls = cg.Clustering(method="dbscan", eps_pct=eps_pct, min_pts=min_pts)
			clusterer = cg.Clustering(method=self.method, outliers=False)
			labels = clusterer.fit_predict(ds)
			ds.set_attr("Clusters", labels, axis=1)
			cg.Merger(min_distance=0.2).merge(ds)
			ds.close()
