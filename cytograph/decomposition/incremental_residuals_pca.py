from typing import Tuple
import loompy
import numpy as np
from sklearn.decomposition import IncrementalPCA
import logging


class IncrementalResidualsPCA():
	"""
	Project a dataset into a reduced feature space using PCA on Pearson residuals.
	See https://www.biorxiv.org/content/10.1101/2020.12.01.405886v1.full.pdf
	"""
	def __init__(self, n_factors: int = 50, **kwargs) -> None:
		"""
		Args:
			n_factors:  	The number of retained components
		"""
		self.n_factors = n_factors

	def fit(self, ds: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray]:
		logging.info(" ResidualsPCA: Loading gene and cell totals")
		totals = ds.ca.TotalUMI
		gene_totals = np.sum(ds[ds.ra.Selected == 1, :], axis=1)
		overall_totals = ds.ca.TotalUMI.sum()

		batch_size = 100_000
		logging.info(f" ResidualsPCA: Fitting PCA on Pearson residuals incrementally in batches of {batch_size:,} cells")
		pca = IncrementalPCA(n_components=self.n_factors)
		for ix in range(0, ds.shape[1], batch_size):
			data = ds[ds.ra.Selected == 1, ix:ix + batch_size].T
			expected = totals[ix:ix + batch_size, None] @ (gene_totals[None, :] / overall_totals)
			residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
			n_cells = residuals.shape[0]
			# residuals = np.clip(residuals, -np.sqrt(n_cells), np.sqrt(n_cells))
			pca.partial_fit(residuals)

		logging.info(f" ResidualsPCA: Transforming residuals incrementally in batches of {batch_size:,} cells")
		factors = np.zeros((ds.shape[1], self.n_factors), dtype="float32")
		for ix in range(0, ds.shape[1], batch_size):
			data = ds[ds.ra.Selected == 1, ix:ix + batch_size].T
			expected = totals[ix:ix + batch_size, None] @ (gene_totals[None, :] / overall_totals)
			residuals = (data - expected) / np.sqrt(expected + np.power(expected, 2) / 100)
			n_cells = residuals.shape[0]
			# residuals = np.clip(residuals, -np.sqrt(n_cells), np.sqrt(n_cells))
			factors[ix:ix + batch_size] = pca.transform(residuals)

		loadings = pca.components_.T
		loadings_all = np.zeros_like(loadings, shape=(ds.shape[0], self.n_factors))
		loadings_all[ds.ra.Selected == 1] = loadings
		return factors