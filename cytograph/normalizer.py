from typing import *
import loompy
import numpy as np
import logging
import luigi

## TODO: this needs to be moved in the project specific repo
class normalizer(luigi.Config):
	level = luigi.IntParameter(default=5000)


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
		# Adjust total count per cell to 10,000
		vals = vals / (self.totals[cells] + 1) * normalizer().level

		# Log transform
		vals = np.log(vals + 1)
		# Subtract mean per gene
		vals = vals - self.mu[:, None]
		if self.standardize:
			# Scale to unit standard deviation per gene
			vals = div0(vals.T, self.sd).T
		return vals

	def fit_transform(self, ds: loompy.LoomConnection, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
		self.fit(ds)
		return self.transform(vals, cells)


def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide(a, b)
		c[~np.isfinite(c)] = 0  # -inf inf NaN
	return c