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


class ClusterLayoutDev(luigi.Task):
	"""
	Luigi Task to cluster a Loom file by Louvain-Jaccard, and perform SFDP layout
	"""
	lineage = luigi.Parameter(default="Ectodermal")  # Alternativelly Endomesodermal
	target = luigi.Parameter(default="All")  # one between Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral, ForebrainVentrothalamic, Midbrain, Hindbrain
	time = luigi.Parameter(default="E7-E18")

	def requires(self) -> luigi.Task:
		return cg.SplitAndPoolAa(lineage=self.lineage, target=self.target, time=self.time)

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".LJ.loom"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.LJ.loom" % (self.lineage, self.target, self.time)))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			logging.info("Creating temporary copy of the input file")
			copyfile(self.input().fn, out_file)
			ds = loompy.connect(out_file)
			cg.cluster_layout(ds)
