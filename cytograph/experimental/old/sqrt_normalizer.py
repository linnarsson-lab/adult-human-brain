from typing import *
import loompy
import numpy as np
import logging
import luigi


class SqrtNormalizer:
	"""
	Normalize a dataset using the square root transform
	"""
	def __init__(self) -> None:
		self.sd = None  # type: np.ndarray
		self.mu = None  # type: np.ndarray
		self.totals = None  # type: np.ndarray

	def fit(self, ds: loompy.LoomConnection, mu: np.ndarray = None, sd: np.ndarray = None, totals: np.ndarray = None) -> None:
		self.sd = sd
		self.mu = mu
		self.totals = totals

		if mu is None or sd is None:
			def std_sqrt(x: np.ndarray) -> float:
				return np.std(np.sqrt(x))

			def mean_sqrt(x: np.ndarray) -> float:
				return np.mean(np.sqrt(x))

			(self.sd, self.mu) = ds.map([std_sqrt, mean_sqrt], axis=0)
		if totals is None:
			self.totals = ds.map([np.sum], chunksize=100, axis=1)[0]

	def transform(self, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
		"""
		Normalize a matrix using the previously calculated aggregate statistics

		Args:
			vals (ndarray):		Matrix of shape (n_genes, n_cells)
			cells (ndarray):	Optional indices of the cells that are represented in vals

		Returns:
			vals_adjusted (ndarray):	The normalized values
		"""
		# Adjust total count per cell to 5,000
		vals = vals / (self.totals[cells] + 1) * 5000

		# Sqrt transform
		vals = np.sqrt(vals)
		# Standardize
		vals = div0(vals, self.sd[:, None])
		# Subtract mean per gene
		vals = vals - self.mu[:, None]
		return vals

	def fit_transform(self, ds: loompy.LoomConnection, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
		self.fit(ds)
		return self.transform(vals, cells)
