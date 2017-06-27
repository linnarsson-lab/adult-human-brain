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


class ClusterL1(luigi.Task):
	"""
	Level 1 clustering
	"""
	tissue = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return [
			cg.PrepareTissuePool(tissue=self.tissue),
			cg.ManifoldL1(tissue=self.tissue)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L1_" + self.tissue + ".clusters.txt"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			cls = cg.Clustering(method=cg.cluster().method)
			labels = cls.fit_predict(ds)
			ds.set_attr("Clusters", labels, axis=1)
			n_labels = np.max(labels) + 1
			with open(out_file, "w") as f:
				f.write(str(n_labels) + " clusters\n")
			logging.info(str(n_labels) + " clusters")
			ds.close()
