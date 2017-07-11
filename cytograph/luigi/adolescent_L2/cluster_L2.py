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


class ClusterL2(luigi.Task):
	"""
	Level 2 clustering of the adolescent dataset
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	method = luigi.Parameter(default='dbscan')  # or 'hdbscan'
	n_genes = luigi.IntParameter(default=1000)
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
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			accessions = None  # type: np.ndarray
			for clustered in self.input():
				ds = loompy.connect(clustered.fn)
				logging.info("Split/pool from " + clustered.fn)
				labels = ds.col_attrs["Class"]

				# Keep track of the gene order in the first file
				if accessions is None:
					accessions = ds.row_attrs["Accession"]

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
				ordering = np.where(ds.row_attrs["Accession"][None, :] == accessions[:, None])[1]
				for (ix, selection, vals) in ds.batch_scan(cells=np.array(cells), axis=1, batch_size=cg.memory().axis1):
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][selection]
					if dsout is None:
						dsout = loompy.create(out_file, vals[ordering, :], ds.row_attrs, ca)
					else:
						dsout.add_columns(vals[ordering, :], ca)
			dsout.close()

			logging.info("Learning the manifold")
			ds = loompy.connect(out_file)
			ml = cg.ManifoldLearning(self.n_genes, self.gtsne, self.alpha)
			(knn, mknn, tsne) = ml.fit(ds)
			ds.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			ds.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			ds.set_attr("_X", tsne[:, 0], axis=1)
			ds.set_attr("_Y", tsne[:, 1], axis=1)

			logging.info("Clustering on the manifold")
			cls = cg.Clustering(method=self.method)
			labels = cls.fit_predict(ds)
			ds.set_attr("Clusters", labels, axis=1)
			n_labels = np.max(labels) + 1

			ds.close()
