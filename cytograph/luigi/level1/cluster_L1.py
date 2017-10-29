from typing import *
import os
from shutil import copyfile
import numpy as np
#import logging
import luigi
import gc
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


class ClusterL1(luigi.Task):
	"""
	Level 1 clustering
	"""
	tissue = luigi.Parameter()
	n_genes = luigi.IntParameter(default=1000)
	gtsne = luigi.BoolParameter(default=True)
	alpha = luigi.FloatParameter(default=1)
	filter_cellcycle = luigi.Parameter(default=None)
	layer = luigi.Parameter(default=None)

	def requires(self) -> luigi.Task:
		return cg.PrepareTissuePool(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L1_" + self.tissue + ".loom"))

	def run(self) -> None:
		logging = cg.logging(self)
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			dsout: loompy.LoomConnection = None
			logging.info("Removing invalid cells")
			for (ix, selection, vals) in ds.batch_scan_layers(cells=np.where(ds.col_attrs["_Valid"] == 1)[0], layers=ds.layer.keys(), batch_size=cg.memory().axis1, axis=1):
				ca = {key: val[selection] for key, val in ds.col_attrs.items()}
				if dsout is None:
					# NOTE Loompy Create should support multilayer !!!!
					if type(vals) is dict:
						dsout = loompy.create(out_file, vals[""], row_attrs=ds.row_attrs, col_attrs=ca, dtype=vals[""].dtype)
						for layername, layervalues in vals.items():
							if layername != "":
								dsout.set_layer(layername, layervalues, dtype=layervalues.dtype)
						dsout = loompy.connect(out_file)
					else:
						loompy.create(out_file, vals, row_attrs=ds.row_attrs, col_attrs=ca)
						dsout = loompy.connect(out_file)
				else:
					dsout.add_columns(vals, ca)
			# dsout.close() causes an exception; disabling gc fixes it. See https://github.com/h5py/h5py/issues/888
			gc.disable()
			dsout.close()
			gc.enable()

			logging.info("Learning the manifold")
			ds = loompy.connect(out_file)
			ml = cg.ManifoldLearning2(n_genes=self.n_genes, gtsne=self.gtsne, alpha=self.alpha, filter_cellcycle=self.filter_cellcycle, layer=self.layer)
			(knn, mknn, tsne) = ml.fit(ds)
			ds.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			ds.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			ds.set_attr("_X", tsne[:, 0], axis=1)
			ds.set_attr("_Y", tsne[:, 1], axis=1)

			logging.info("Clustering on the manifold")
			cls = cg.Clustering(method="mknn_louvain", min_pts=10)
			labels = cls.fit_predict(ds)
			ds.set_attr("Clusters", labels, axis=1)
			logging.info(f"Found {labels.max() + 1} clusters")
			cg.Merger(min_distance=0.2).merge(ds)
			logging.info(f"Merged to {ds.col_attrs['Clusters'].max() + 1} clusters")
			ds.close()

			# dsout = loompy.connect(out_file)
			# ml = cg.ManifoldLearning(n_genes=self.n_genes, gtsne=self.gtsne, alpha=self.alpha, filter_cellcycle=self.filter_cellcycle, layer=self.layer)
			# (knn, mknn, tsne) = ml.fit(dsout)

			# dsout.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			# dsout.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			# dsout.set_attr("_X", tsne[:, 0], axis=1)
			# dsout.set_attr("_Y", tsne[:, 1], axis=1)

			# min_pts = 10
			# eps_pct = 90
			# if self.tissue == "Sympathetic":
			# 	min_pts = 10
			# 	eps_pct = 80
			# cls = cg.Clustering(method=cg.cluster().method, outliers=not cg.cluster().no_outliers, min_pts=min_pts, eps_pct=eps_pct)
			# labels = cls.fit_predict(dsout)
			# dsout.set_attr("Clusters", labels, axis=1)
			# n_labels = np.max(labels) + 1
			# dsout.close()
