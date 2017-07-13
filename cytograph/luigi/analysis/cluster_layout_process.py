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


class ClusterLayoutProcess(luigi.Task):
	"""
	Luigi Task to cluster a Loom file by Louvain-Jaccard, and perform SFDP layout
	"""
	processname = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.StudyProcessPool(processname=self.processname)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "%s.LJ.loom" % self.processname))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			logging.info("Creating temporary copy of the input file")
			copyfile(self.input().fn, out_file)
			ds = loompy.connect(out_file)
			cg.cluster_layout(ds)
