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
	def __init__(self, n_genes: int = 1000, n_factors: int = 100, k: int = 10, k_smoothing: int = 100) -> None:
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_smoothing = k_smoothing
		self.k = k

	def fit(self, ds: loompy.LoomConnection) -> None:
		# Select genes
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		self.genes = genes
		data = ds.sparse(rows=genes).T

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)

		# KNN in HPF space
		logging.info(f"Computing KNN (k={self.k_smoothing}) in latent space")
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T  # Normalize so the sums are one because JSD requires it
		logging.info("Fitting a ball tree index")
		nn = cg.BalancedKNN(self.k_smoothing, metric=cg.jensen_shannon_distance, maxl=2 * self.k_smoothing, sight_k=2 * self.k_smoothing)
		nn.fit(theta)
		logging.info("Finding nearest neighbors")
		knn = nn.kneighbors_graph(theta, mode='connectivity')
		knn.setdiag(1)

		# Poisson smoothing (in place)
		logging.info(f"Poisson smoothing")
		ds["smoothened"] = 'int32'
		if "spliced" in ds.layers:
			ds["spliced_ps"] = 'int32'
			ds["unspliced_ps"] = 'int32'
		for (ix, indexes, view) in ds.scan(axis=0):
			if "spliced" in ds.layers:
				ds["spliced_ps"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced_ps"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["smoothened"][indexes.min(): indexes.max() + 1, :] = ds["spliced_ps"][indexes.min(): indexes.max() + 1, :] + ds["unspliced_ps"][indexes.min(): indexes.max() + 1, :]
			else:
				ds["smoothened"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T

		# Select genes
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False, layer="smoothened")
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes, layer="smoothened").fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		data = ds["smoothened"].sparse(rows=genes).T

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		# Here we normalize so the sums over components are one, because JSD requires it
		# and because the components will be exactly proportional to cell size
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T
		beta_all = np.zeros((ds.shape[0], hpf.beta.shape[1]))
		beta_all[genes] = hpf.beta
		ds.ra.HPF = beta_all
		ds.ca.HPF = theta

		logging.info(f"tSNE embedding from latent space")
		tsne = TSNE(n_components=2, metric=cg.jensen_shannon_distance).fit_transform(theta)
		ds.ca.TSNE = tsne

		logging.info(f"Computing balanced KNN (k = {self.k}) in latent space")
		bnn = cg.BalancedKNN(k=self.k, metric=cg.jensen_shannon_distance, maxl=2 * self.k, sight_k=2 * self.k)
		bnn.fit(theta)
		knn = bnn.kneighbors_graph(mode='distance')
		mknn = knn.minimum(knn.transpose())
		# Convert distances to similarities
		knn.data = 1 - knn.data
		mknn.data = 1 - mknn.data
		ds.col_graphs.KNN = knn
		ds.col_graphs.MKNN = mknn

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain()
		labels = pl.fit_predict(ds, "KNN")
		ds.ca.Clusters = labels + 1
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {labels.max() + 1} clusters")
