from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import scipy.cluster.hierarchy as hierarchy
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from polo import optimal_leaf_ordering
import scipy.stats
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt


class Aggregator:
	def __init__(self, *, n_markers: int = 10, f: Union[float, List[float]] = 0.2) -> None:
		self.n_markers = n_markers
		self.f = f

	def aggregate(self, ds: loompy.LoomConnection, out_file: str, agg_spec: Dict[str, str] = None) -> None:
		if agg_spec is None:
			agg_spec = {
				"Age": "tally",
				"Clusters": "first",
				"Class": "mode",
				"_Total": "mean",
				"Sex": "tally",
				"Tissue": "tally",
				"SampleID": "tally",
				"TissuePool": "first",
				"Outliers": "mean"
			}
		cells = ds.col_attrs["Clusters"] >= 0
		labels = ds.col_attrs["Clusters"][cells]
		n_labels = len(set(labels))

		logging.info("Aggregating clusters by mean")
		cg.aggregate_loom(ds, out_file, None, "Clusters", "mean", agg_spec)
		with loompy.connect(out_file) as dsout:
			logging.info("Trinarizing")
			if type(self.f) is list or type(self.f) is tuple:
				for ix, f in enumerate(self.f):
					trinaries = cg.Trinarizer(f=f).fit(ds)
					if ix == 0:
						dsout.layers["trinaries"] = trinaries
					else:
						dsout.layers[f"trinaries_{f}"] = trinaries
			else:
				trinaries = cg.Trinarizer(f=self.f).fit(ds)
				dsout.layers["trinaries"] = trinaries

			logging.info("Computing cluster gene enrichment scores")
			(markers, enrichment, qvals) = cg.MarkerSelection(self.n_markers).fit(ds)
			dsout.layers["enrichment"] = enrichment
			dsout.layers["enrichment_q"] = qvals

			dsout.ca.NCells = np.bincount(labels, minlength=n_labels)

			# Renumber the clusters
			logging.info("Renumbering clusters by similarity, and permuting columns")
			if "_Selected" in ds.ra:
				genes = (ds.ra._Selected == 1)
			else:
				logging.info("Normalization")
				normalizer = cg.Normalizer(False)
				normalizer.fit(ds)
				logging.info("Selecting up to 1000 genes")
				genes = cg.FeatureSelection(1000).fit(ds, mu=normalizer.mu, sd=normalizer.sd)

			data = np.log(dsout[:, :] + 1)[genes, :].T
			D = pdist(data, 'euclidean')
			Z = hc.linkage(D, 'ward')
			optimal_Z = optimal_leaf_ordering(Z, D)
			ordering = hc.leaves_list(optimal_Z)

			# Permute the aggregated file, and renumber
			dsout.permute(ordering, axis=1)
			dsout.ca.Clusters = np.arange(n_labels)

			# Renumber the original file, and permute
			d = dict(zip(ordering, np.arange(n_labels)))
			new_clusters = np.array([d[x] if x in d else -1 for x in ds.ca.Clusters])
			ds.ca.Clusters = new_clusters
			ds.permute(np.argsort(ds.col_attrs["Clusters"]), axis=1)

			# Reorder the genes, markers first, ordered by enrichment in clusters
			logging.info("Permuting rows")
			mask = np.zeros(ds.shape[0], dtype=bool)
			mask[markers] = True
			# fetch enrichment from the aggregated file, so we get it already permuted on the column axis
			gene_order = np.zeros(ds.shape[0], dtype='int')
			gene_order[mask] = np.argmax(dsout.layer["enrichment"][mask, :], axis=1)
			gene_order[~mask] = np.argmax(dsout.layer["enrichment"][~mask, :], axis=1) + dsout.shape[1]
			gene_order = np.argsort(gene_order)
			ds.permute(gene_order, axis=0)
			dsout.permute(gene_order, axis=0)

			data = trinaries[:, ordering][gene_order, :][:self.n_markers * n_labels, :].T
			cluster_scores = []
			for ix in range(n_labels):
				cluster_scores.append(data[ix, ix * 10:(ix + 1) * 10].sum())
			dsout.ca.ClusterScore = np.array(cluster_scores)


def aggregate_loom(ds: loompy.LoomConnection, out_file: str, select: np.ndarray, group_by: str, aggr_by: str, aggr_ca_by: Dict[str, str], return_matrix: bool = False) -> np.ndarray:
	"""
	Aggregate a Loom file by applying aggregation functions to the main matrix as well as to the column attributes

	Args:
		ds			The Loom file
		out_file	The name of the output Loom file (will be appended to if it exists)
		select		Bool array giving the columns to include (or None, to include all)
		group_by	The column attribute to group by
		aggr_by 	The aggregation function for the main matrix
		aggr_ca_by	The aggregation functions for the column attributes (or None to skip)

	Remarks:
		aggr_by gives the aggregation function for the main matrix
		aggr_ca_by is a dictionary with column attributes as keys and aggregation functionas as values
		
		Aggregation functions can be any valid aggregation function from here: https://github.com/ml31415/numpy-groupies

		In addition, you can specify:
			"tally" to count the number of occurences of each value of a categorical attribute
	"""
	ca = {}  # type: Dict[str, np.ndarray]
	if select is not None:
		raise ValueError("The 'select' argument is deprecated")
	labels = (ds.ca[group_by]).astype('int')
	_, zero_strt_sort_noholes_lbls = np.unique(labels, return_inverse=True)
	n_groups = len(set(labels))
	if aggr_ca_by is not None:
		for key in ds.col_attrs.keys():
			if key not in aggr_ca_by:
				continue
			func = aggr_ca_by[key]
			if func == "tally":
				for val in set(ds.col_attrs[key]):
					ca[key + "_" + val] = npg.aggregate(zero_strt_sort_noholes_lbls, (ds.col_attrs[key] == val).astype('int'), func="sum", fill_value=0)
			elif func == "mode":
				def mode(x):
					return scipy.stats.mode(x)[0][0]
				ca[key] = npg.aggregate(zero_strt_sort_noholes_lbls, ds.col_attrs[key], func=mode, fill_value=0).astype('str')
			elif func == "mean":
				ca[key] = npg.aggregate(zero_strt_sort_noholes_lbls, ds.col_attrs[key], func=func, fill_value=0)
			elif func == "first":
				ca[key] = npg.aggregate(zero_strt_sort_noholes_lbls, ds.col_attrs[key], func=func, fill_value=ds.col_attrs[key][0])

	m = np.empty((ds.shape[0], n_groups))
	for (_, selection, view) in ds.scan(axis=0):
		vals_aggr = npg.aggregate(zero_strt_sort_noholes_lbls, view[:, :], func=aggr_by, axis=1, fill_value=0)
		m[selection, :] = vals_aggr

	if return_matrix:
		return m

	loompy.create_append(out_file, m, ds.ra, ca, fill_values="auto")
