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


# Config classes should be camel cased
class clustering(luigi.Config):
	n_genes = luigi.IntParameter(default=2000)
	standardize = luigi.BoolParameter(default=False)
	n_components = luigi.IntParameter(default=50)
	use_ica = luigi.BoolParameter(default=True)
	k = luigi.IntParameter(default=30)
	lj_resolution = luigi.FloatParameter(default=1.0)


class ClusterLayoutL2(luigi.Task):
	"""
	Luigi Task to cluster a Loom file by Louvain-Jaccard, and perform SFDP layout
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.SplitAndPool(tissue=self.tissue, major_class=self.major_class, project=self.project)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".LJ.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			logging.info("Creating temporary copy of the input file")
			copyfile(self.input().fn, out_file)
			ds = loompy.connect(out_file)
			cg.cluster_layout(ds)
