from typing import List

import numpy as np
import pandas as pd
from harmony import harmonize
from scipy.stats import ks_2samp
from sklearn.decomposition import IncrementalPCA

import loompy
from cytograph.preprocessing import Normalizer


class PCA:
	"""
	Project a dataset into a reduced feature space using incremental PCA. The projection can be fit
	to one dataset then used to project another. To work properly, both datasets must be normalized in the same
	way prior to projection.
	"""
	def __init__(self, genes: np.ndarray, max_n_components: int = 50, layer: str = None, test_significance: bool = True, batch_keys: List[str] = None) -> None:
		"""
		Args:
			genes:				The genes to use for the projection
			max_n_components: 	The maximum number of projected components
			layer:				The layer to use as input
			test_significance:	If true, return only a subset of up to max_n_components that are significant
			batch_keys:			Keys (attribute names) to use as batch keys for batch correction, or None to omit batch correction
		"""
		self.genes = genes
		self.n_components = max_n_components
		self.test_significance = test_significance
		self.layer = layer
		self.cells = None  # type: np.ndarray
		self.pca = None  # type: IncrementalPCA
		self.sigs = None  # type: np.ndarray
		self.batch_keys = batch_keys

	def fit(self, ds: loompy.LoomConnection, normalizer: Normalizer, cells: np.ndarray = None) -> None:
		if cells is None:
			cells = np.fromiter(range(ds.shape[1]), dtype='int')

		# Support out-of-order datasets
		key = None
		if "Accession" in ds.row_attrs:
			key = "Accession"

		self.pca = IncrementalPCA(n_components=self.n_components)
		layer = self.layer if self.layer is not None else ""
		for (_, selection, view) in ds.scan(items=cells, axis=1, layers=[layer], key=key):
			if len(selection) < self.n_components:
				continue
			vals = normalizer.transform(view.layers[layer][:, :], selection)
			self.pca.partial_fit(vals[self.genes, :].transpose())		# PCA on the selected genes

	def transform(self, ds: loompy.LoomConnection, normalizer: Normalizer, cells: np.ndarray = None) -> np.ndarray:
		if cells is None:
			cells = np.arange(ds.shape[1])

		transformed = np.zeros((cells.shape[0], self.pca.n_components_))
		j = 0

		# Support out-of-order datasets
		key = None
		if "Accession" in ds.row_attrs:
			key = "Accession"

		layer = self.layer if self.layer is not None else ""
		for (_, selection, view) in ds.scan(items=cells, axis=1, layers=[layer], key=key):
			vals = normalizer.transform(view.layers[layer][:, :], selection)
			n_cells_in_batch = selection.shape[0]
			transformed[j:j + n_cells_in_batch, :] = self.pca.transform(vals[self.genes, :].transpose())
			j += n_cells_in_batch

		if self.test_significance:
			# Must select significant components only once, and reuse for future transformations
			if self.sigs is None:
				pvalue_KS = np.zeros(transformed.shape[1])  # pvalue of each component
				for i in range(1, transformed.shape[1]):
					(_, pvalue_KS[i]) = ks_2samp(transformed[:, i - 1], transformed[:, i])
				self.sigs = np.where(pvalue_KS < 0.1)[0]
				if len(self.sigs) == 0:
					self.sigs = (0, 1)

			transformed = transformed[:, self.sigs]

		if self.batch_keys is not None and len(self.batch_keys) > 0:
			keys_df = pd.DataFrame.from_dict({k: ds.ca[k] for k in self.batch_keys})
			transformed = harmonize(transformed, keys_df, batch_key=self.batch_keys)
		return transformed

	def fit_transform(self, ds: loompy.LoomConnection, normalizer: Normalizer, cells: np.ndarray = None) -> np.ndarray:
		self.fit(ds, normalizer, cells)
		return self.transform(ds, normalizer, cells)
