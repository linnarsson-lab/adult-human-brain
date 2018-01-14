import os
import csv
import numpy as np
import cytograph as cg
from typing import *
from sklearn.decomposition import IncrementalPCA, FastICA
from scipy.stats import ks_2samp
import loompy


class PCAProjection:
	"""
	Project a dataset into a reduced feature space using PCA. The projection can be fit
	to one dataset then used to project another. To work properly, both datasets must be normalized in the same 
	way prior to projection.
	"""
	def __init__(self, genes: np.ndarray, max_n_components: int = 50, layer: str=None, nng: np.ndarray = None) -> None:
		"""
		Args:
			genes:				The genes to use for the projection
			max_n_components: 	The maximum number of projected components
			nng					Non-neuronal genes, to be zeroed in neurons (where TaxonomyRank1 == "Neurons")
		"""
		self.genes = genes
		self.n_components = max_n_components
		self.layer = layer
		self.nng = nng
		self.cells = None  # type: np.ndarray
		self.pca = None  # type: IncrementalPCA
		self.sigs = None  # type: np.ndarray
		self.accessions = None  # type: np.ndarray

	def fit(self, ds: loompy.LoomConnection, normalizer: cg.Normalizer, cells: np.ndarray = None) -> None:
		if cells is None:
			cells = np.fromiter(range(ds.shape[1]), dtype='int')
		n_cells = cells.shape[0]
		n_genes = self.genes.shape[0]

		# Support out-of-order datasets
		if "Accession" in ds.row_attrs:
			self.accessions = ds.row_attrs["Accession"]

		self.pca = IncrementalPCA(n_components=self.n_components)
		if self.layer is not None:
			# NOTE TO AVOID a BUG with layer of pickled objects
			try:
				for (ix, selection, view) in ds.scan(items=cells, axis=1, layers=[self.layer]):
					vals = normalizer.transform(view.layers[self.layer][:, :], selection)
					if self.nng is not None:
						vals[np.where(self.nng)[0][:, None], np.where(ds.TaxonomyRank1 == "Neurons")[0]] = 0
					self.pca.partial_fit(vals[self.genes, :].transpose())		# PCA on the selected genes
			except AttributeError:
				self.layer = None
		if self.layer is None:
			for (ix, selection, view) in ds.scan(items=cells, axis=1):
				vals = normalizer.transform(view[:, :], selection)
				if self.nng is not None:
					vals[np.where(self.nng)[0][:, None], np.where(ds.TaxonomyRank1 == "Neurons")[0]] = 0
				self.pca.partial_fit(vals[self.genes, :].transpose())		# PCA on the selected genes

	def transform(self, ds: loompy.LoomConnection, normalizer: cg.Normalizer, cells: np.ndarray = None) -> np.ndarray:
		if cells is None:
			cells = np.fromiter(range(ds.shape[1]), dtype='int')
		n_cells = cells.shape[0]

		# Support out--of-order datasets
		if self.accessions is not None:
			# This is magic sauce for making the order of one list be like another
			ordering = np.where(ds.row_attrs["Accession"][None, :] == self.accessions[:, None])[1]

		transformed = np.zeros((cells.shape[0], self.pca.n_components_))
		j = 0

		if self.layer is not None:
			# NOTE TO AVOID a BUG with layer of pickled objects
			try:
				for (ix, selection, view) in ds.scan(items=cells, axis=1, layers=[self.layer]):
					vals = normalizer.transform(view.layers[self.layer][:, :], selection)
					if self.nng is not None:
						vals[np.where(self.nng)[0][:, None], np.where(ds.TaxonomyRank1 == "Neurons")[0]] = 0
					n_cells_in_batch = selection.shape[0]
					transformed[j:j + n_cells_in_batch, :] = self.pca.transform(vals[self.genes, :].transpose())
					j += n_cells_in_batch
			except AttributeError:
				self.layer = None
		if self.layer is None:
			for (ix, selection, view) in ds.scan(items=cells, axis=1):
				vals = normalizer.transform(view[:, :], selection)
				if self.nng is not None:
					vals[np.where(self.nng)[0][:, None], np.where(ds.TaxonomyRank1 == "Neurons")[0]] = 0
				n_cells_in_batch = selection.shape[0]
				transformed[j:j + n_cells_in_batch, :] = self.pca.transform(vals[self.genes, :].transpose())
				j += n_cells_in_batch

		# Must select significant components only once, and reuse for future transformations
		if self.sigs is None:
			pvalue_KS = np.zeros(transformed.shape[1])  # pvalue of each component
			for i in range(1, transformed.shape[1]):
				(_, pvalue_KS[i]) = ks_2samp(transformed[:, i - 1], transformed[:, i])
			self.sigs = np.where(pvalue_KS < 0.1)[0]
			if len(self.sigs) == 0:
				self.sigs = (0, 1)

		transformed = transformed[:, self.sigs]

		return transformed

	def fit_transform(self, ds: loompy.LoomConnection, normalizer: cg.Normalizer, cells: np.ndarray = None) -> np.ndarray:
		self.fit(ds, normalizer, cells)
		return self.transform(ds, normalizer, cells)
