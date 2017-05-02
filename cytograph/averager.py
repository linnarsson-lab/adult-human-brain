from math import exp, lgamma, log
import logging
import os
from typing import *
import numpy as np
import loompy
import numpy_groupies.aggregate_numpy as npg


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


class Averager:
	def __init__(self, func: str = "mean") -> None:
		self.func = func

	def calculate_and_save(self, ds: loompy.LoomConnection, output_file: str, aggregator_class: str = "Clusters", category_summary: Tuple = ("Age", "SampleID")) -> None:
		"""Calculate the average table ad save it as a .loom file

		Args
		----
		ds : loompy.LoomConnection
			a loompy connection to the input loom file
		output_file : str
			the path to the output file
		aggregator_class : str
			the column attribute to use as a categorical varaible to aggregate
			defaults to "Clusters"
		category_summary : str
			the other column attributes for which a count of every category will be provided as a separate column attribute in the output file
			defaults to ("Age", "SampleID")

		Returns
		-------
		None

		"""
		cells = np.where(ds.col_attrs["Clusters"] >= 0)[0]
		labels = ds.col_attrs[aggregator_class][cells]
		categories, categories_ix = np.unique(labels, return_inverse=True)
		Ncat = len(categories)
		ca = {aggregator_class: categories, "OriginalFile": np.array([output_file] * Ncat)}
		ra = {"Accession": ds.row_attrs["Accession"], "Gene": ds.row_attrs["Gene"]}
		for category_class in category_summary:
			for unique_element in set(ds.col_attrs[category_class]):
				ca["%s_%s" % (category_class, unique_element)] = npg.aggregate_numba.aggregate(categories_ix, ds.col_attrs[category_class][cells] == unique_element, func="sum")
		m = np.empty((ds.shape[0], Ncat))
		for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=0):
			vals_avg = npg.aggregate_numba.aggregate(categories_ix, vals, func=self.func, axis=1)
			m[selection, :] = vals_avg
		dsout = loompy.create(output_file, m, ra, ca)