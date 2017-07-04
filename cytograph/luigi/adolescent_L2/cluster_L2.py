from typing import *
import os
from shutil import copyfile
import numpy as np
import logging
import luigi
import cytograph as cg
import loompy
import logging
from scipy import sparse
from scipy.special import polygamma
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from scipy.stats import ks_2samp
import networkx as nx
import hdbscan
from sklearn.cluster import DBSCAN


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
		return cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L2_" + self.major_class + "_" + self.tissue + ".clustered.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			logging.info("Learning the manifold")
			ds = loompy.connect(self.input().fn)
			ml = cg.ManifoldLearning(self.n_genes, self.gtsne, self.alpha)
			(knn, mknn, tsne) = ml.fit(ds)
			ds.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			ds.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			ds.set_attr("_X", tsne[:, 0], axis=1)
			ds.set_attr("_Y", tsne[:, 1], axis=1)

			logging.info("Clustering on the manifold")
			ds = loompy.connect(self.input().fn)
			cls = cg.Clustering(method=self.method)
			labels = cls.fit_predict(ds)
			ds.set_attr("Clusters", labels, axis=1)
			n_labels = np.max(labels) + 1

			logging.info("Removing outliers")
			cells = np.where(ds.col_attrs["Outliers"] == 0)[0]
			outlier_label = ds.Clusters[ds.Outliers == 1][0]
			new_labels = np.array([x if x < outlier_label else x - 1 for x in ds.Clusters])[cells]
			dsout = None  # type: loompy.LoomConnection
			for (ix, selection, vals) in ds.batch_scan(cells=cells, axis=1):
				ca = {key: v[selection] for key, v in ds.col_attrs.items()}
				if dsout is None:
					dsout = loompy.create(out_file, vals, ds.row_attrs, ca)
				else:
					dsout.add_columns(vals, ca)
			dsout.set_attr("Clusters", new_labels, axis=1)
			ds.close()

			# Close and reopen because of some subtle bug in assigning and reading back col attrs
			dsout.close()
			dsout = loompy.connect(out_file)
			logging.info("Relearning the manifold with outliers removed")
			ml = cg.ManifoldLearning(self.n_genes, self.gtsne, self.alpha)
			logging.info(dsout.col_attrs["_Valid"] == 1)
			(knn, mknn, tsne) = ml.fit(dsout)

			dsout.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			dsout.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			dsout.set_attr("_X", tsne[:, 0], axis=1)
			dsout.set_attr("_Y", tsne[:, 1], axis=1)
