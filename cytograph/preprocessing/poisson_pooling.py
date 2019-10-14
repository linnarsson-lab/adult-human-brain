import logging

import numpy as np
import scipy.sparse as sparse
from pynndescent import NNDescent

import loompy
from cytograph.decomposition import HPF, identify_technical_factors
from cytograph.enrichment import FeatureSelectionByVariance
from cytograph.metrics import jensen_shannon_distance

from .normalizer import Normalizer


class PoissonPooling:
	def __init__(self, k_pooling: int = 10, n_genes: int = 2000, n_factors: int = 96, mask: np.ndarray = None, remove_technical_factors: bool = False, compute_velocity: bool = False, n_threads: int = 0):
		self.k_pooling = k_pooling
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.mask = mask
		self.remove_technical_factors = remove_technical_factors
		self.compute_velocity = compute_velocity
		self.n_threads = n_threads

		self.knn: sparse.coo_matrix = None  # Make this available after fitting in case it's useful downstream

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.debug(f"Selecting {self.n_genes} genes")
		normalizer = Normalizer(False)
		normalizer.fit(ds)
		genes = FeatureSelectionByVariance(self.n_genes, mask=self.mask).fit(ds)
		self.genes = genes
		data = ds.sparse(rows=genes).T

		# Subsample to lowest number of UMIs
		if "TotalUMI" in ds.ca:
			totals = ds.ca.TotalUMI
		else:
			totals = ds.map([np.sum], axis=1)[0]
		min_umis = int(np.min(totals))
		logging.debug(f"Subsampling to {min_umis} UMIs")
		temp = data.toarray()
		for c in range(temp.shape[0]):
			temp[c, :] = np.random.binomial(temp[c, :].astype('int32'), min_umis / totals[c])
		data = sparse.coo_matrix(temp)

		# HPF factorization
		hpf = HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False, n_threads=self.n_threads)
		hpf.fit(data)
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T  # Normalize so the sums are one because JSD requires it

		if self.remove_technical_factors and "Batch" in ds.ca and "Replicate" in ds.ca:
			technical = identify_technical_factors(theta, ds.ca.Batch, ds.ca.Replicate)
			logging.debug(f"Removing {technical.sum()} technical factors")
			theta = theta[:, ~technical]

		# KNN in HPF space
		logging.debug(f"Computing KNN (k={self.k_pooling}) in latent space")
		nn = NNDescent(data=theta, metric=jensen_shannon_distance)
		indices, distances = nn.query(theta, k=self.k_pooling)
		# Note: we convert distances to similarities here, to support Poisson smoothing below
		knn = sparse.csr_matrix(
			(1 - np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])), (theta.shape[0], theta.shape[0])
		)
		knn.setdiag(1)
		self.knn = knn

	def fit_transform(self, ds: loompy.LoomConnection) -> None:
		# Poisson pooling
		self.fit(ds)
		knn = self.knn
		logging.debug(f"Poisson pooling")
		ds["pooled"] = 'int32'
		if self.compute_velocity and "spliced" in ds.layers:
			ds["spliced_pooled"] = 'int32'
			ds["unspliced_pooled"] = 'int32'
			for (ix, indexes, view) in ds.scan(axis=0, layers=["spliced", "unspliced"], what=["layers"]):
				ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] + ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :]
		else:
			for (ix, indexes, view) in ds.scan(axis=0, layers=[""], what=["layers"]):
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T
