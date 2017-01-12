import os
from typing import *
import logging
from shutil import copyfile
import numpy as np
import loompy
import differentiation_topology as dt
import luigi
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
from differentiation_topology.pipeline import QualityControl, AutoAnnotation
from sklearn.svm import SVR


class Subset(luigi.Task):
	"""
	Luigi Task to make a subset of a Loom file based on auto-annotation
	"""
	build_folder = luigi.Parameter(default="")
	name = luigi.Parameter()
	tag = luigi.Parameter()

	def output(self) -> luigi.LocalTarget:
		return luigi.LocalTarget(os.path.join(self.build_folder, self.name + "_" + self.tag + ".loom"))

	def requires(self) -> Any:
		return AutoAnnotation(self.name)

	def run(self) -> None:
		aa = {}
		with self.requires().open("r") as f:
			for line in f.readlines():
				items = line.split("\t")
				aa[items[0]] = np.array([float(f) for f in items[1:]])

		clusters = np.where(aa[self.tag] > 0.5)[0]

		# TODO: extract only the cells that have the right cluster labels
