from math import exp, lgamma, log
import logging
from typing import *
import numpy as np
import loompy
import numpy_groupies as npg


class Averager:
	def __init__(self, func: str = "mean") -> None:
		self.func = func

	def calculate_and_save(self, ds: loompy.LoomConnection, output_file: str, age_stats: bool = True) -> None:
		cells = np.where(ds.col_attrs["_Valid"] == 1)[0]
		labels = ds.col_attrs["Clusters"][cells]
		Nclust = np.max(labels) + 1
		ca = {"Cluster": np.arange(Nclust), "OriginalFile": np.array([output_file] * Nclust)}
		ra = {"Accession": ds.row_attrs["Accession"], "Gene": ds.row_attrs["Gene"]}
		if age_stats:
			def parse_age(age: str) -> float:
				unit, amount = age[0], float(age[1:])
				if unit == "P":
					amount += 19.
				return amount
			number_of_days = np.fromiter(map(parse_age, ds.col_attrs["Age"]), dtype=float)
			ca["AgeAverage"] = npg.aggregate_numba.aggregate(labels, number_of_days[cells], func="mean")
			ca["AgeStd"] = npg.aggregate_numba.aggregate(labels, number_of_days[cells], func="std")
			ca["Age05thPercentile"] = npg.aggregate_numpy.aggregate(labels, number_of_days[cells], func=lambda x: np.percentile(x, 5, interpolation="linear"))
			ca["Age25thPercentile"] = npg.aggregate_numpy.aggregate(labels, number_of_days[cells], func=lambda x: np.percentile(x, 25, interpolation="linear"))
			ca["Age75thPercentile"] = npg.aggregate_numpy.aggregate(labels, number_of_days[cells], func=lambda x: np.percentile(x, 50, interpolation="linear"))
			ca["Age95thPercentile"] = npg.aggregate_numpy.aggregate(labels, number_of_days[cells], func=lambda x: np.percentile(x, 95, interpolation="linear"))

			for age in set(ds.Age):
				ca["N_cells_%s" % age] = npg.aggregate_numba.aggregate(labels, ds.col_attrs["Age"][cells] == age, func="sum")


		m = np.empty((ds.shape[0], Nclust))
		for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=0):
			vals_avg = npg.aggregate_numba.aggregate(labels, vals, func=self.func, axis=1)
			m[selection, :] = vals_avg
		dsout = loompy.create(output_file, m, ra, ca)