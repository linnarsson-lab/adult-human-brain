from math import exp, lgamma, log
import logging
from typing import *
import numpy as np
import loompy
import numpy_groupies as npg


class Averager:
	def __init__(self, func: str = "mean") -> None:
		self.func = func

	def calculate_and_save(self, ds: loompy.LoomConnection, output_file: str) -> None:
		cells = np.where(ds.col_attrs["_Valid"] == 1)[0]
		labels = ds.col_attrs["Clusters"][cells]
		Nclust = np.max(labels) + 1
		ca = {"Cluster": np.arange(Nclust), "OriginalFile": np.array([output_file] * Nclust)}
		ra = {"Accession": ds.row_attrs["Accession"], "Gene": ds.row_attrs["Gene"]}
		m = np.empty((ds.shape[0], Nclust))
		for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=0):
			vals_avg = npg.aggregate_numba.aggregate(labels, vals, func=self.func, axis=1)
			m[selection, :] = vals_avg
		dsout = loompy.create(output_file, m, ra, ca)