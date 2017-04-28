import logging
import numpy as np
from typing import *
from sklearn.svm import SVR
import loompy
import cytograph as cg


class MarkerSelection:
	def __init__(self, n_markers: int, labels_attr: str = "Clusters") -> None:
		self.n_markers = n_markers
		self.labels_attr = labels_attr

	def fit(self, ds: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Finds n_markers genes per cluster using enrichment score

		Args:
			ds (LoomConnection):	Dataset

		Returns:
			ndarray of selected genes (list of ints)
		"""
		labels = ds.col_attrs[self.labels_attr]
		cells = labels >= 0
		labels = labels[cells]
		n_labels = max(labels) + 1

		sizes = np.bincount(labels, minlength=n_labels)
		nnz = cg.aggregate_loom(ds, None, cells, "Clusters", np.count_nonzero, None, return_matrix=True)
		means = cg.aggregate_loom(ds, None, cells, "Clusters", "mean", None, return_matrix=True)
		(nnz_overall, means_overall) = ds.map([np.count_nonzero, np.mean], axis=0, selection=cells)
		f_nnz = nnz / sizes
		f_nnz_overall = nnz_overall / len(cells)
		enrichment = (f_nnz + 0.1) / (f_nnz_overall[None].T + 0.1) * (means + 0.01) / (means_overall[None].T + 0.01)

		# Select best markers
		included = []  # type: List[int]
		excluded = set(np.where(ds.row_attrs["_Valid"] == 0)[0])  # type: Set[int]
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
