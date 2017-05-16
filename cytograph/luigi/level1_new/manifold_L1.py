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


class ManifoldL1(luigi.Task):
	"""
	Luigi Task to learn the high-dimensional manifold and embed it as a multiscale KNN graph, as well as t-SNE projection
	"""
	tissue = luigi.Parameter()
	n_genes = luigi.IntParameter(default=1000)
	gtsne = luigi.BoolParameter(default=True)

	def requires(self) -> luigi.Task:
		return cg.PrepareTissuePool(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.tissue + ".L1.manifold.txt"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)

			ml = cg.ManifoldLearning(self.n_genes, self.gtsne)
			(knn, mknn, tsne) = ml.fit(ds)

			ds.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			ds.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			ds.set_attr("_X", tsne[:, 0], axis=1)
			ds.set_attr("_Y", tsne[:, 1], axis=1)

			with open(out_file, "w") as f:
				f.write("This file is just a placeholder\n")
			ds.close()
