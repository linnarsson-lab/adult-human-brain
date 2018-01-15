import logging
import numpy as np
from typing import *
from sklearn.svm import SVR
from statsmodels.sandbox.stats.multicomp import multipletests
import loompy
import cytograph as cg


class MarkerSelection:
	def __init__(self, n_markers: int, labels_attr: str = "Clusters", mask: np.ndarray=None) -> None:
		self.n_markers = n_markers
		self.labels_attr = labels_attr
		self.alpha = 0.1
		self.mask = mask

	def fit(self, ds: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Finds n_markers genes per cluster using enrichment score

		Args:
			ds (LoomConnection):	Dataset

		Returns:
			ndarray of selected genes (list of ints)
			ndarray of enrichment scores
			ndarray of FDR-corrected P values (i.e. q values)
		"""
		# First compute the null distribution using permutation test
		logging.info("Computing enrichment null distribution")
		labels = ds.col_attrs[self.labels_attr]
		ds.set_attr(self.labels_attr, np.random.permutation(labels), axis=1)
		try:
			(_, null_enrichment) = self._fit(ds)
		except:
			raise
		finally:
			ds.set_attr(self.labels_attr, labels, axis=1)

		# Get the observed enrichment statistics
		logging.info("Computing enrichment statistic")
		(selected, enrichment) = self._fit(ds)

		# Calculate FDR-corrected P values
		logging.info("Computing enrichment FDR-corrected P values")
		qvals = np.zeros_like(enrichment)
		for ix in range(enrichment.shape[1]):
			null_values = null_enrichment[:, ix]
			null_values.sort()
			values = enrichment[:, ix]
			pvals = 1 - np.searchsorted(null_values, values) / values.shape[0]
			(_, q, _, _) = multipletests(pvals, self.alpha, method="fdr_bh")
			qvals[:, ix] = q

		return (selected, enrichment, qvals)

	def _fit(self, ds: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Finds n_markers genes per cluster using enrichment score

		Args:
			ds (LoomConnection):	Dataset

		Returns:
			ndarray of selected genes (list of ints)
			ndarray of enrichment scores
		"""
		labels = ds.col_attrs[self.labels_attr]
		cells = labels >= 0
		labels = labels[cells]
		n_labels = np.max(labels) + 1
		n_cells = cells.sum()

		# Number of cells per cluster
		sizes = np.bincount(labels, minlength=n_labels)
		# Number of nonzero values per cluster
		nnz = cg.aggregate_loom(ds, None, cells, self.labels_attr, np.count_nonzero, None, return_matrix=True)
		# Mean value per cluster
		means = cg.aggregate_loom(ds, None, cells, self.labels_attr, "mean", None, return_matrix=True)
		# Non-zeros and means over all cells
		(nnz_overall, means_overall) = ds.map([np.count_nonzero, np.mean], axis=0, selection=cells)
		# Scale by number of cells
		f_nnz = nnz / sizes
		f_nnz_overall = nnz_overall / len(cells)

		# Means and fraction non-zero values in other clusters (per cluster)
		means_other = ((means_overall * n_cells)[None].T - (means * sizes)) / (n_cells - sizes)
		f_nnz_other = ((f_nnz_overall * n_cells)[None].T - (f_nnz * sizes)) / (n_cells - sizes)

		# enrichment = (f_nnz + 0.1) / (f_nnz_overall[None].T + 0.1) * (means + 0.01) / (means_overall[None].T + 0.01)
		enrichment = (f_nnz + 0.1) / (f_nnz_other + 0.1) * (means + 0.01) / (means_other + 0.01)

		# Select best markers
		included = []  # type: List[int]
		if self.mask is None:
			excluded = set(np.where(ds.row_attrs["_Valid"] == 0)[0])  # type: Set[int]
		else:
			excluded = set(np.where(np.logical_and(ds.row_attrs["_Valid"] == 0, self.mask))[0])
		for ix in range(max(labels) + 1):
			enriched = np.argsort(enrichment[:, ix])[::-1]
			n = 0
			count = 0
			while count < self.n_markers:
				if enriched[n] in excluded:
					n += 1
					continue
				included.append(enriched[n])
				excluded.add(enriched[n])
				n += 1
				count += 1
		return (np.array(included), enrichment)
