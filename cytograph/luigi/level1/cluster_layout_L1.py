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


class ClusterLayoutL1(luigi.Task):
	"""
	Luigi Task to cluster a Loom file by Louvain-Jaccard, and perform SFDP layout
	"""
	tissue = luigi.Parameter()
	use_magic = luigi.BoolParameter(default=False)

	def requires(self) -> luigi.Task:
		return cg.PrepareTissuePool(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, self.tissue + ".LJ.loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			if self.use_magic:
				logging.info("Imputing expression values using MAGIC")
				ds = loompy.connect(self.input().fn)
				cg.magic_imputation(ds, out_file)
				ds.close()
			else:
				logging.info("Creating temporary copy of the input file")
				copyfile(self.input().fn, out_file)
			ds = loompy.connect(out_file)
			cg.cluster_layout(ds, gtsne=True)
