import numpy as np
from typing import *
import loompy
import logging
import scipy.sparse as sparse
from pynndescent import NNDescent
from cytograph.species import Species
from .normalizer import Normalizer
from cytograph.enrichment import FeatureSelection
from cytograph.decomposition import HPF, identify_technical_factors
from cython.metric import jensen_shannon_distance


class PoissonPooling:
	def __init__(self, k_pooling: int = 10, n_genes: int = 2000, n_factors: int = 96, mask_cell_cycle: bool = False, remove_technical_factors: bool = False, compute_velocity: bool = False):
		self.k_pooling = k_pooling
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.mask_cell_cycle = mask_cell_cycle
		self.remove_technical_factors = remove_technical_factors
		self.compute_velocity = compute_velocity

	def poisson_pooling(self, ds: loompy.LoomConnection) -> None:
		cc_genes = Species(ds).cell_cycle_genes
		n_samples = ds.shape[1]
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = Normalizer(False)
		normalizer.fit(ds)
		mask = None
		if self.mask_cell_cycle:
			mask = np.isin(ds.ra.Gene, cc_genes)
		genes = FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd, mask=mask)
		self.genes = genes
		data = ds.sparse(rows=genes).T

		# Subsample to lowest number of UMIs
		# TODO: figure out how to do this without making the data matrix dense
		if "TotalRNA" not in ds.ca:
			(ds.ca.TotalRNA, ) = ds.map([np.sum], axis=1)
		totals = ds.ca.TotalRNA
		min_umis = np.min(totals)
		logging.info(f"Subsampling to {min_umis} UMIs")
		temp = data.toarray()
		for c in range(temp.shape[0]):
			temp[c, :] = np.random.binomial(temp[c, :].astype('int32'), min_umis / totals[c])
		data = sparse.coo_matrix(temp)

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T  # Normalize so the sums are one because JSD requires it

		if self.remove_technical_factors and "Batch" in ds.ca and "Replicate" in ds.ca:
			technical = identify_technical_factors(theta, ds.ca.Batch, ds.ca.Replicate)
			logging.info(f"Removing {technical.sum()} technical factors")
			theta = theta[:, ~technical]
		else:
			logging.warn("Could not analyze technical factors because attributes 'Batch' and 'Replicate' are missing")

		# KNN in HPF space
		logging.info(f"Computing KNN (k={self.k_pooling}) in latent space")
		nn = NNDescent(data=theta, metric=jensen_shannon_distance)
		indices, distances = nn.query(theta, k=self.k_pooling)
		# Note: we convert distances to similarities here, to support Poisson smoothing below
		knn = sparse.csr_matrix(
			(1 - np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])), 		(theta.shape[0], theta.shape[0])
		)
		knn.setdiag(1)

		# Poisson pooling
		logging.info(f"Poisson pooling")
		ds["pooled"] = 'int32'
		if self.compute_velocity and "spliced" in ds.layers:
			ds["spliced_pooled"] = 'int32'
			ds["unspliced_pooled"] = 'int32'
			for (ix, indexes, view) in ds.scan(axis=0, layers=["spliced", "unspliced"]):
				ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] + ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :]
		else:
			for (ix, indexes, view) in ds.scan(axis=0, layers=["pooled"]):
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T
