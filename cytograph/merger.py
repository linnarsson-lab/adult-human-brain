from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
import scipy.cluster.hierarchy as hierarchy
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from polo import optimal_leaf_ordering
import scipy.stats
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt


class Merger:
	"""
	Merge clusters based on a cluster distance metric
	"""
	def __init__(self, n_markers: int = 10, min_distance: float = 3, pep: float = 0.1) -> None:
		self.n_markers = n_markers
		self.min_distance = min_distance
		self.pep = pep

	def merge(self, ds: loompy.LoomConnection) -> None:
		cells = ds.col_attrs["Clusters"] >= 0
		labels = ds.col_attrs["Clusters"][cells]
		n_labels = len(set(labels))

		logging.info("Merging similar clusters")
		logging.info("Trinarizing")
		trinaries = cg.Trinarizer().fit(ds)

		logging.info("Computing cluster gene enrichment scores")
		(markers, enrichment, qvals) = cg.MarkerSelection(self.n_markers).fit(ds)
		
		def discordance_distance(a: np.ndarray, b: np.ndarray) -> float:
			"""
			Number of genes that are discordant with given PEP, divided by number of clusters
			"""
			return np.sum((1 - a) * b + a * (1 - b) > 1 - self.pep) / n_labels

		data = trinaries[markers, :].T
		Z = hc.linkage(data, 'complete', metric=discordance_distance)
		D = pdist(data, discordance_distance)
		optimal_Z = optimal_leaf_ordering(Z, D)
		ordering = hc.leaves_list(optimal_Z)
		merged = hc.fcluster(optimal_Z, self.min_distance, criterion='distance') - 1

		# Keep the outliers separate
		if "Outliers" in ds.col_attrs.keys() and np.any(ds.col_attrs["Outliers"] == 1):
			outliers = ds.col_attrs["Clusters"][ds.col_attrs["Outliers"] == 1][0]
			if (merged == outliers).sum() > 1:
				merged[outliers] = -1
				merged += 1

		# Renumber the clusters
		d = dict(zip(np.arange(n_labels), merged))
		new_clusters = np.array([d[x] if x in d else -1 for x in ds.Clusters])
		ds.set_attr("Clusters", new_clusters, axis=1)
		logging.info(f"Merged {n_labels} -> {len(set(new_clusters))} clusters")

		if not len(set(ds.Clusters)) == ds.Clusters.max() + 1:
			raise ValueError("There are holes in the cluster ID sequence!")
