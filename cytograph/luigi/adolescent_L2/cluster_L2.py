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


class ClusterL2(luigi.Task):
	"""
	Level 2 clustering of the adolescent dataset
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	n_genes = luigi.IntParameter(default=1000)
	gtsne = luigi.BoolParameter(default=True)
	method = luigi.Parameter(default='louvain')  # or 'hdbscan'

	def requires(self) -> luigi.Task:
		return [
			cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class, project=self.project), 
			cg.ManifoldL2(tissue=self.tissue, major_class=self.major_class, project="Adolescent")
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".clusters.txt"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)

			n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
			n_total = ds.shape[1]
			logging.info("%d of %d cells were valid", n_valid, n_total)
			logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])
			cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

			if self.method == "hdbscan":
				logging.info("HDBSCAN clustering in t-SNE space")
				tsne_pos = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[cells, :]
				clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
				labels = clusterer.fit_predict(tsne_pos)
				labels_all = np.zeros(ds.shape[1], dtype='int') + -1
				labels_all[cells] = labels
				ds.set_attr("Clusters", labels_all, axis=1)
			else:
				logging.info("Louvain clustering on the multiscale KNN graph")
				(a, b, w) = ds.get_edges("KNN", axis=1)
				knn = sparse.coo_matrix((w, (a, b)), shape=(ds.shape[1], ds.shape[1])).tocsr()[cells, :][:, cells]
				lj = cg.LouvainJaccard(resolution=10, jaccard=False)
				labels = lj.fit_predict(knn.tocoo())
				# Make labels for excluded cells == -1
				labels_all = np.zeros(ds.shape[1], dtype='int') + -1
				labels_all[cells] = labels
				ds.set_attr("Clusters", labels_all, axis=1)
			n_labels = np.max(labels) + 1
			with open(out_file, "w") as f:
				f.write(str(n_labels) + " clusters\n")
			logging.info(str(n_labels) + " clusters")
			ds.close()

