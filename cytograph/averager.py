from math import exp, lgamma, log
import logging
import os
from typing import *
import numpy as np
import loompy
import numpy_groupies.aggregate_numpy as npg


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