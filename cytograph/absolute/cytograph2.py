import cytograph as cg
import numpy as np
import scipy.sparse as sparse
import loompy
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, NearestNeighbors
import logging
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from typing import *
import os


class Cytograph2:
	def __init__(self, n_genes: int = 8000, n_factors: int = 64, k: int = 25, k_smoothing: int = 5, outliers: bool = False) -> None:
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_smoothing = k_smoothing
		self.k = k
		self.outliers = outliers

	def fit(self, ds: loompy.LoomConnection) -> None:
		# Select genes
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		self.genes = genes
		data = ds.sparse(rows=genes).T
		n_samples = data.shape[0]

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)

		# KNN in HPF space
		logging.info(f"Computing KNN (k={self.k_smoothing}) in latent space")
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T  # Normalize so the sums are one because JSD requires it
		nn = cg.BallTreeJS(data=theta)
		(distances, indices) = nn.query(theta, k=self.k_smoothing)
		# Note: we convert distances to similarities here, to support Poisson smoothing below
		knn = sparse.csr_matrix(
			(1 - np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])),
			(theta.shape[0], theta.shape[0])
		)
		knn.setdiag(1)

		# Poisson smoothing (in place)
		logging.info(f"Poisson pooling")
		ds["pooled"] = 'int32'
		if "spliced" in ds.layers:
			ds["spliced_pp"] = 'int32'
			ds["unspliced_pp"] = 'int32'
		for (ix, indexes, view) in ds.scan(axis=0):
			if "spliced" in ds.layers:
				ds["spliced_pp"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced_pp"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = ds["spliced_pp"][indexes.min(): indexes.max() + 1, :] + ds["unspliced_pp"][indexes.min(): indexes.max() + 1, :]
			else:
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T

		# Select genes
		logging.info(f"Selecting {self.n_genes} genes after smoothing")
		normalizer = cg.Normalizer(False, layer="pooled")
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes, layer="pooled").fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		data = ds["pooled"].sparse(rows=genes).T

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		# Here we normalize so the sums over components are one, because JSD requires it
		# and because otherwise the components will be exactly proportional to cell size
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T
		beta_all = np.zeros((ds.shape[0], hpf.beta.shape[1]))
		beta_all[genes] = (hpf.beta.T / hpf.beta.sum(axis=1)).T
		ds.ra.HPF = beta_all
		ds.ca.HPF = theta

		# Expected values
		# TODO: Calculate expectations for spliced and unspliced too
		logging.info(f"Computing expected values")
		ds["expected"] = 'float32'  # Create a layer of floats
		start = 0
		batch_size = 6400
		temp = np.zeros((ds.shape[0], batch_size))
		while start < n_samples:
			# Compute PPV for the genes that were used for HPF, and slot them into the full gene list
			if hpf.theta.shape[0] - start < batch_size:  # For the last window, temp needs to be a little smaller
				temp = np.zeros((ds.shape[0], hpf.theta.shape[0] - start))
			temp[genes] = (hpf.theta[start: start + batch_size, :] @ hpf.beta.T).T
			# Assign the batch to the full matrix at the right slot
			ds["expected"][:, start: start + batch_size] = temp
			start += batch_size

		logging.info(f"tSNE embedding from latent space")
		tsne = cg.tsne_js(theta)
		ds.ca.TSNE = tsne

		logging.info(f"Computing balanced KNN (k = {self.k}) in latent space")
		bnn = cg.BalancedKNN(k=self.k, metric="js", maxl=2 * self.k, sight_k=2 * self.k, n_jobs=-1)
		bnn.fit(theta)
		knn = bnn.kneighbors_graph(mode='distance')
		mknn = knn.minimum(knn.transpose())
		# Convert distances to similarities
		knn.data = 1 - knn.data
		mknn.data = 1 - mknn.data
		ds.col_graphs.KNN = knn
		ds.col_graphs.MKNN = mknn

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain(outliers=self.outliers)
		labels = pl.fit_predict(ds, "KNN")
		ds.ca.Clusters = labels + min(labels)
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {labels.max() + 1} clusters")
