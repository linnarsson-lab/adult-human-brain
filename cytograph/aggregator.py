from typing import *
import os
import csv
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi
import scipy.cluster.hierarchy as hierarchy
import numpy_groupies.aggregate_numpy as npg
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from polo import optimal_leaf_ordering


class Aggregator:
	def __init__(self, n_markers: int = 10) -> None:
		self.n_markers = n_markers
	
	def aggregate(self, ds: loompy.LoomConnection, out_file: str) -> None:
		ca_aggr = {
			"Age": "tally",
			"Clusters": "first",
			"Class": "first",
			"Class_Astrocyte": "mean",
			"Class_Cycling": "mean",
			"Class_Ependymal": "mean",
			"Class_Neurons": "mean",
			"Class_Immune": "mean",
			"Class_Oligos": "mean",
			"Class_OEC": "mean",
			"Class_Schwann": "mean",
			"Class_Vascular": "mean",
			"_Total": "mean",
			"Sex": "tally",
			"Tissue": "tally",
			"SampleID": "tally",
			"TissuePool": "first"
		}
		cells = ds.col_attrs["Clusters"] >= 0
		labels = ds.col_attrs["Clusters"][cells]
		n_labels = len(set(labels))

		logging.info("Aggregating clusters by mean")
		cg.aggregate_loom(ds, out_file, cells, "Clusters", "mean", ca_aggr)
		dsout = loompy.connect(out_file)

		logging.info("Trinarizing")
		trinaries = cg.Trinarizer().fit(ds)
		dsout.set_layer("trinaries", trinaries)

		logging.info("Computing cluster gene enrichment scores")
		(markers, enrichment) = cg.MarkerSelection(self.n_markers).fit(ds)
		dsout.set_layer("enrichment", enrichment)
		top_genes = np.argsort(np.max(enrichment, axis=1))[:1000]

		dsout.set_attr("NCells", np.bincount(labels, minlength=n_labels), axis=1)

		# Renumber the clusters
		logging.info("Renumbering clusters by similarity, and permuting columns")
		data = np.log(dsout[:, :] + 1)[ds.row_attrs["_Selected"] == 1, :].T
		D = pdist(data, 'euclidean')
		Z = hc.linkage(D, 'ward')
		optimal_Z = optimal_leaf_ordering(Z, D)
		ordering = hc.leaves_list(optimal_Z)
		# Permute the aggregated file, and renumber
		dsout.permute(ordering, axis=1)
		dsout.set_attr("Clusters", np.arange(n_labels), axis=1)
		# Renumber the original file, and permute
		new_clusters = renumber(ds.col_attrs["Clusters"], ordering, np.arange(n_labels))
		ds.set_attr("Clusters", new_clusters, axis=1)
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


def renumber(a: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
	ordering = np.argsort(keys)
	keys = keys[ordering]
	values = values[ordering]
	index = np.digitize(a.ravel(), keys, right=True)
	return(values[index].reshape(a.shape))


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
		cols = np.where(select)[0]
	else:
		cols = np.fromiter(range(ds.shape[1]))
	labels = (ds.col_attrs[group_by][cols]).astype('int')
	n_groups = len(set(labels))
	if aggr_ca_by is not None:
		for key in ds.col_attrs.keys():
			if key not in aggr_ca_by:
				continue
			func = aggr_ca_by[key]
			if func == "tally":
				for val in set(ds.col_attrs[key]):
					ca[key + "_" + val] = npg.aggregate(labels, ds.col_attrs[key][cols] == val, func="sum")
			else:
				ca[key] = npg.aggregate(labels, ds.col_attrs[key][cols], func=func, fill_value=ds.col_attrs[key][cols][0])
	m = np.empty((ds.shape[0], n_groups))
	for (ix, selection, vals) in ds.batch_scan(cells=cols, genes=None, axis=0):
		vals_aggr = npg.aggregate(labels, vals, func=aggr_by, axis=1)
		m[selection, :] = vals_aggr

	if return_matrix:
		return m

	if os.path.exists(out_file):
		dsout = loompy.connect(out_file)
		dsout.add_columns(m, ca, fill_values="auto")
		dsout.close()
	else:
		loompy.create(out_file, m, ds.row_attrs, ca)
