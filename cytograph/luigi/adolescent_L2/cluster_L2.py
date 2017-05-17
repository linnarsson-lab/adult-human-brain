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

	def requires(self) -> luigi.Task:
		return [
			cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class),
			cg.ManifoldL2(tissue=self.tissue, major_class=self.major_class)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "L2_" + self.major_class + "_" + self.tissue + ".clusters.txt"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			cls = cg.Clustering(method=self.method)
			labels = cls.fit_predict(ds)
			ds.set_attr("Clusters", labels, axis=1)
			n_labels = np.max(labels) + 1
			with open(out_file, "w") as f:
				f.write(str(n_labels) + " clusters\n")
			logging.info(str(n_labels) + " clusters")
			ds.close()

