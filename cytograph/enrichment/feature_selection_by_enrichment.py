import logging
from typing import Tuple, List

import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy import sparse
import loompy


class FeatureSelectionByEnrichment:
	def __init__(self, n_markers: int = 10, mask: np.ndarray = None, labels_attr: str = "Clusters", findq: bool = True) -> None:
		self.n_markers = n_markers
		self.labels_attr = labels_attr
		self.alpha = 0.1
		self.mask = mask
		self.findq = findq
		self.enrichment: np.ndarray = None
		self.qvals: np.ndarray = None

	def fit(self, ds: loompy.LoomConnection) -> np.ndarray:
		"""
		Finds n_markers genes per cluster using enrichment score

		Args:
			ds (LoomConnection):	Dataset

		Returns:
			ndarray of selected genes (list of ints)
			ndarray of enrichment scores
			ndarray of FDR-corrected P values (i.e. q values)
		"""
		# Get the observed enrichment statistics
		logging.info("Computing enrichment statistic")
		(genes, self.enrichment) = self._fit(ds)

		if self.findq:
			# Compute the null distribution using permutation test
			logging.info("Computing enrichment null distribution")
			labels = ds.col_attrs[self.labels_attr]
			ds.ca[self.labels_attr] = np.random.permutation(labels)
			(_, null_enrichment) = self._fit(ds)
			ds.ca[self.labels_attr] = labels

			# Calculate FDR-corrected P values
			logging.info("Computing enrichment FDR-corrected P values")
			self.qvals = np.zeros_like(self.enrichment)
			for ix in range(self.enrichment.shape[1]):
				null_values = null_enrichment[:, ix]
				null_values.sort()
				values = self.enrichment[:, ix]
				pvals = 1 - np.searchsorted(null_values, values) / values.shape[0]
				(_, q, _, _) = multipletests(pvals, self.alpha, method="fdr_bh")
				self.qvals[:, ix] = q

		selected = np.zeros(ds.shape[0], dtype=bool)
		selected[np.sort(genes)] = True
		return selected

	def select(self, ds: loompy.LoomConnection) -> np.ndarray:
		selected = self.fit(ds)
		ds.ra.Selected = selected.astype("int")
		return selected

	def _fit(self, ds: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray]:
		labels = ds.ca[self.labels_attr]
		n_labels = max(labels) + 1
		n_cells = ds.shape[1]

		# reorder matrix
		mtx = ds.sparse().tocsr()
		ordering = np.argsort(labels)
		arr_list = []
		chunksize = 100000000 // mtx.shape[1]
		start = 0
		while start < mtx.shape[0]:
			submatrix = mtx[start:start + chunksize, :]
			arr_list.append(submatrix[:, ordering])
			start = start + chunksize
		mtx = sparse.vstack(arr_list)

		# split matrix on label indices
		ix = [np.where(np.sort(labels) == x)[0][0] for x in range(n_labels)] + [n_cells]
		arr_list = []
		for i in range(len(ix) - 1):
			arr_list.append(mtx[:, ix[i]:ix[i + 1]])

		# Number of cells per cluster
		sizes = np.zeros(n_labels)
		# Number of nonzero values per cluster
		nnz = np.zeros((ds.shape[0], n_labels))
		# Mean value per cluster
		means = np.zeros((ds.shape[0], n_labels))

		for i, arr in enumerate(arr_list):
			nnz[:, i] = arr.getnnz(axis=1)
			means[:, i] = np.squeeze((arr.mean(axis=1).A))
			sizes[i] = arr.shape[1]

		# Non-zeros and means over all cells
		mtx = sparse.hstack(arr_list)
		nnz_overall = mtx.getnnz(axis=1)
		means_overall = np.squeeze((mtx.mean(axis=1).A))

		# Scale by number of cells
		n_cells = mtx.shape[1]
		f_nnz = nnz / sizes
		f_nnz_overall = nnz_overall / n_cells

		# Means and fraction non-zero values in other clusters (per cluster)
		means_other = ((means_overall * n_cells)[None].T - (means * sizes)) / (n_cells - sizes)
		f_nnz_other = ((f_nnz_overall * n_cells)[None].T - (f_nnz * sizes)) / (n_cells - sizes)

		# enrichment = (f_nnz + 0.1) / (f_nnz_overall[None].T + 0.1) * (means + 0.01) / (means_overall[None].T + 0.01)
		enrichment = (f_nnz + 0.1) / (f_nnz_other + 0.1) * (means + 0.01) / (means_other + 0.01)

		# Select best markers
		if "Valid" not in ds.ra:
			logging.info("Recomputing the list of valid genes")
			nnz = ds.map([np.count_nonzero], axis=0)[0]
			valid_genes = np.logical_and(nnz > 10, nnz < ds.shape[1] * 0.6)
			ds.ra.Valid = valid_genes.astype('int')

		included: List[int] = []

		if self.mask is None:
			excluded = set(np.where(ds.ra.Valid == 0)[0])
		else:
			excluded = set(np.where(np.logical_or(ds.ra.Valid == 0, self.mask))[0])

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
