import numpy as np
from typing import *
import scipy.sparse as sparse
import loompy
import logging
from numba import jit
from tqdm import tqdm


class NeighborhoodEnrichment:
	def __init__(self, epsilon: float = 0.1) -> None:
		self.epsilon = epsilon
	
	def fit(self, X: np.ndarray, knn: sparse.csr_matrix, k: int) -> np.ndarray:
		"""
		Args:
			X 		Input vector of expression values, shape=(n_cells)
			knn		KNN connectivity matrix, shape=(n_cells, n_cells)
		
		Remarks:
			knn is assumed to have a single k for all cells (if not, the enrichment will
			be only approximate).
		"""
		n_cells = X.shape[0]
		nonzeros = (X > 0).astype('int')
		nz_bycell = knn.multiply(nonzeros).sum(axis=1)
		sum_bycell = knn.multiply(X).sum(axis=1)

		total_nz = nonzeros.sum()
		total_sum = X.sum()

		nz_enrichment = (nz_bycell / k + self.epsilon) / ((total_nz - nz_bycell) / (n_cells - k) + self.epsilon)
		mean_enrichment = (sum_bycell / k + self.epsilon) / ((total_sum - sum_bycell) / (n_cells - k) + self.epsilon)

		return (nz_enrichment.A * mean_enrichment.A).T[0]

	def fit_loom(self, ds: loompy.LoomConnection, *, tolayer: str = "enrichment", knn: Union[str, sparse.csr_matrix] = "KNN") -> None:
		if tolayer not in ds.layers:
			ds[tolayer] = "float32"
		if type(knn) is str:
			knn_matrix = ds.col_graphs[knn].tocsr()
		else:
			knn_matrix = knn
		k = knn_matrix.count_nonzero() / knn_matrix.shape[0]
		with tqdm(total=ds.shape[0], desc="Neighborhood enrichment") as pbar:
			for ix, selection, view in ds.scan(axis=0):
				for j in range(view.shape[0]):
					ds[tolayer][j + ix, :] = self.fit(view[j, :], knn_matrix, k)
				pbar.update(view.shape[0])
