from typing import *
import loompy
import numpy as np


class Normalizer:
	"""
	Normalize and optionally standardize a dataset, dealing properly with edge cases such as division by zero.
	"""
	def __init__(self, standardize: bool = False) -> None:
		self.standardize = standardize
		self.sd = None  # type: np.ndarray
		self.mu = None  # type: np.ndarray
		self.totals = None  # type: np.ndarray

	def fit(self, ds: loompy.LoomConnection, mu: np.ndarray = None, sd: np.ndarray = None, totals: np.ndarray = None) -> None:
		self.sd = sd
		self.mu = mu
		self.totals = totals

		if mu is None or sd is None:
			(self.sd, self.mu) = ds.map([np.std, np.mean], axis=0)
		if totals is None:
			self.totals = ds.map(np.sum, axis=1)

	def transform(self, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
		"""
		Normalize a matrix using the previously calculated aggregate statistics

		Args:
			vals (ndarray):		Matrix of shape (n_genes, n_cells)
			cells (ndarray):	Optional indices of the cells that are represented in vals

		Returns:
			vals_adjusted (ndarray):	The normalized values
		"""
		# Adjust total count per cell to 10,000
		vals = vals / (self.totals[cells] + 1) * 10000

		# Log transform
		vals = np.log(vals + 1)
		# Subtract mean per gene
		vals = vals - self.mu[:, None]
		if self.standardize:
			# Scale to unit standard deviation per gene
			vals = self._div0(vals, self.sd[:, None])
		return vals

	def fit_transform(self, ds: loompy.LoomConnection, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
		self.fit(ds)
		return self.transform(vals, cells)

	def _div0(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
		""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
		with np.errstate(divide='ignore', invalid='ignore'):
			c = np.true_divide(a, b)
			c[~np.isfinite(c)] = 0  # -inf inf NaN
		return c