import logging
import warnings
from typing import List

import numpy as np
import scipy.sparse as sparse
from numba.core.errors import NumbaPendingDeprecationWarning, NumbaPerformanceWarning
from pynndescent import NNDescent
from scipy.sparse import SparseEfficiencyWarning

import loompy
from cytograph.decomposition import HPF, PCA
from cytograph.enrichment import FeatureSelectionByVariance
from cytograph.metrics import jensen_shannon_distance
from cytograph.preprocessing import Normalizer


class PoissonPooling:
	def __init__(self, k_pooling: int = 10, n_genes: int = 2000, n_factors: int = 96, mask: np.ndarray = None, compute_velocity: bool = False, n_threads: int = 0, factorization: str = "HPF", batch_keys: List[str] = None):
		self.k_pooling = k_pooling
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.mask = mask
		self.compute_velocity = compute_velocity
		self.n_threads = n_threads
		self.factorization = factorization
		self.batch_keys = batch_keys
		if batch_keys == []:
			self.batch_keys = None
	
		self.knn: sparse.coo_matrix = None  # Make this available after fitting in case it's useful downstream

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info(f"Normalizing and selecting {self.n_genes} genes")
		normalizer = Normalizer(False)
		normalizer.fit(ds)
		genes = FeatureSelectionByVariance(self.n_genes, mask=self.mask).fit(ds)
		self.genes = genes

		if self.factorization == 'PCA' or self.factorization == 'both' or self.batch_keys is not None:
			factorization = "PCA"
		else:
			factorization = "HPF"
		
		if factorization == "PCA":
			n_components = min(50, ds.shape[1])
			logging.info("PCA projection to %d components", n_components)
			pca = PCA(genes, max_n_components=self.n_factors, test_significance=False, batch_keys=self.batch_keys)
			transformed = pca.fit_transform(ds, normalizer)
		else:
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
			n_components = min(self.n_factors, round(ds.shape[1]/3))
			hpf = HPF(k=n_components, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False, n_threads=self.n_threads)
			hpf.fit(data)
			transformed = (hpf.theta.T / hpf.theta.sum(axis=1)).T  # Normalize so the sums are one because JSD requires it

		# KNN in latent space
		logging.info(f"Computing KNN (k={self.k_pooling}) in latent space")
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=NumbaPerformanceWarning)  # Suppress warnings about numba not being able to parallelize code
			warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)  # Suppress warnings about future deprecations
			warnings.simplefilter("ignore", category=SparseEfficiencyWarning)  # Suppress warnings about setting the diagonal to 1
			nn = NNDescent(data=transformed, metric=(jensen_shannon_distance if factorization == "HPF" else "euclidean"))
			indices, distances = nn.query(transformed, k=self.k_pooling)
			# Note: we convert distances to similarities here, to support Poisson smoothing below
			knn = sparse.csr_matrix(
				(np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])), (transformed.shape[0], transformed.shape[0])
			)
			max_d = knn.data.max()
			knn.data = (max_d - knn.data) / max_d
			knn.setdiag(1)  # This causes a sparse efficiency warning, but it's not a slow step relative to everything else
			self.knn = knn

	def fit_transform(self, ds: loompy.LoomConnection) -> None:
		# Poisson pooling
		self.fit(ds)
		knn = self.knn.astype("bool")

		logging.debug(f"Poisson pooling")
		ds["pooled"] = 'int32'
		if self.compute_velocity and "spliced" in ds.layers:
			ds["spliced_pooled"] = 'int32'
			ds["unspliced_pooled"] = 'int32'
			for (_, indexes, view) in ds.scan(axis=0, layers=["spliced", "unspliced"], what=["layers"]):
				ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] = view.layers["spliced"][:, :] @ knn.T
				ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :] = view.layers["unspliced"][:, :] @ knn.T
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] + ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :]
		else:
			for (_, indexes, view) in ds.scan(axis=0, layers=[""], what=["layers"]):
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = view[:, :] @ knn.T
