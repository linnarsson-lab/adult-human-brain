import cytograph as cg
import numpy as np
import scipy.sparse as sparse
import loompy
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, NearestNeighbors
import logging
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import normalize
from typing import *
import os
from .velocity_inference import fit_gamma


def mkl_bug() -> None:
	x = np.arange(1000000).reshape(1000, 1000)
	assert np.std(x, axis=1)[0] == np.std(x[0, :]), "Numpy is broken (MKL parallel bug)"


class Cytograph2:
	def __init__(self, *, n_genes: int = 8000, n_factors: int = 64, k: int = 25, k_pooling: int = 5, outliers: bool = False, radius: float = 0.8, required_genes: str = None, timestep: float = 1) -> None:
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_pooling = k_pooling
		self.k = k
		self.outliers = outliers
		self.radius = radius
		self.required_genes = required_genes
		self.timestep = timestep

	def fit(self, ds: loompy.LoomConnection) -> None:
		mkl_bug()
		# Select genes
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		self.genes = genes
		data = ds.sparse(rows=genes).T
		n_samples = data.shape[0]

		mkl_bug()		
		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)

		mkl_bug()
		# KNN in HPF space
		logging.info(f"Computing KNN (k={self.k_pooling}) in latent space")
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T  # Normalize so the sums are one because JSD requires it
		nn = cg.BallTreeJS(data=theta)
		(distances, indices) = nn.query(theta, k=self.k_pooling)
		# Note: we convert distances to similarities here, to support Poisson smoothing below
		knn = sparse.csr_matrix(
			(1 - np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])),
			(theta.shape[0], theta.shape[0])
		)
		knn.setdiag(1)

		mkl_bug()
		# Poisson smoothing (in place, except the main layer)
		logging.info(f"Poisson pooling")
		ds["pooled"] = 'int32'
		for (ix, indexes, view) in ds.scan(axis=0):
			if "spliced" in ds.layers:
				ds["spliced"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = ds["spliced"][indexes.min(): indexes.max() + 1, :] + ds["unspliced"][indexes.min(): indexes.max() + 1, :]
			else:
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T

		mkl_bug()
		# Select genes
		logging.info(f"Selecting {self.n_genes} genes after smoothing")
		normalizer = cg.Normalizer(False, layer="pooled")
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes, layer="pooled").fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		# Make sure to include these genes
		genes = np.union1d(genes, np.where(np.isin(ds.ra.Gene, self.required_genes))[0])
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		data = ds["pooled"].sparse(rows=genes).T

		mkl_bug()
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

		mkl_bug()
		# HPF factorization of spliced/unspliced
		if "spliced" in ds.layers:
			logging.info(f"HPF of spliced molecules")
			data_spliced = ds["spliced"].sparse(rows=genes).T
			theta_spliced = hpf.transform(data_spliced)
			theta_spliced = (theta_spliced.T / theta_spliced.sum(axis=1)).T
			logging.info(f"HPF of unspliced molecules")
			data_unspliced = ds["unspliced"].sparse(rows=genes).T
			theta_unspliced = hpf.transform(data_unspliced)
			theta_unspliced = (theta_unspliced.T / theta_unspliced.sum(axis=1)).T

		mkl_bug()
		# Expected values
		# TODO: Calculate expectations for spliced and unspliced too
		logging.info(f"Computing expected values")
		ds["expected"] = 'float32'  # Create a layer of floats
		start = 0
		batch_size = 6400
		temp = np.zeros((ds.shape[0], batch_size))
		if "spliced" in ds.layers:
			ds["spliced_exp"] = 'float32'
			ds['unspliced_exp'] = 'float32'
			temp_spliced = np.zeros((ds.shape[0], batch_size))
			temp_unspliced = np.zeros((ds.shape[0], batch_size))
		while start < n_samples:
			# Compute PPV for the genes that were used for HPF, and slot them into the full gene list
			if theta.shape[0] - start < batch_size:  # For the last window, temp needs to be a little smaller
				temp = np.zeros((ds.shape[0], theta.shape[0] - start))
				temp_spliced = np.zeros((ds.shape[0], theta.shape[0] - start))
				temp_unspliced = np.zeros((ds.shape[0], theta.shape[0] - start))
			# Cumpute and assign the batch to the full matrix at the right slot
			temp[genes] = (theta[start: start + batch_size, :] @ hpf.beta.T).T
			ds["expected"][:, start: start + batch_size] = temp
			if "spliced" in ds.layers:
				temp_spliced[genes] = (theta_spliced[start: start + batch_size, :] @ hpf.beta.T).T
				ds["spliced_exp"][:, start: start + batch_size] = temp
				temp_unspliced[genes] = (theta_unspliced[start: start + batch_size, :] @ hpf.beta.T).T
				ds["unspliced_exp"][:, start: start + batch_size] = temp				
			start += batch_size

		mkl_bug()
		logging.info(f"2D tSNE embedding from latent space")
		tsne = cg.tsne_js(theta, radius=1 - self.radius)
		ds.ca.TSNE = tsne

		mkl_bug()
		logging.info(f"3D tSNE embedding from latent space")
		tsne = cg.tsne_js(theta, n_components=3, radius=1 - self.radius)  # Note: tsne_js is defined in terms of distance, not similarity
		ds.ca.TSNE3D = tsne

		mkl_bug()
		logging.info(f"2D UMAP embedding from latent space")
		umap_embedder = UMAP(metric="cosine", spread=2, repulsion_strength=2, n_neighbors=50, n_components=2)
		umap = umap_embedder.fit_transform(theta)
		ds.ca.UMAP = umap

		mkl_bug()
		logging.info(f"3D UMAP embedding from latent space")
		umap3d = UMAP(metric="cosine", spread=2, repulsion_strength=2, n_neighbors=50, n_components=3).fit_transform(theta)
		ds.ca.UMAP3D = umap3d

		mkl_bug()
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
		knn.setdiag(0)
		knn = knn.tocoo()
		inside = knn.data > self.radius
		rnn = sparse.coo_matrix((knn.data[inside], (knn.row[inside], knn.col[inside])), shape=knn.shape)
		ds.col_graphs.RNN = rnn

		mkl_bug()
		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain(outliers=self.outliers)
		labels = pl.fit_predict(ds, graph="RNN", embedding="HPF")
		ds.ca.Clusters = labels + min(labels)
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {labels.max() + 1} clusters")
		mkl_bug()

		logging.info("Fitting gamma for velocity inference")
		selected = ds.ra.Selected == 1
		n_genes = ds.shape[0]
		s = ds["spliced"][selected, :]
		u = ds["unspliced"][selected, :]
		gamma = fit_gamma(s, u)
		gamma_all = np.zeros(n_genes)
		gamma_all[selected] = gamma
		ds.ra.Gamma = gamma_all

		logging.info("Extrapolating future expression")
		velocity = u + gamma[:, None] * s
		future = s + velocity * self.timestep
		np.clip(future, 0, None, out=future)
		future = np.random.poisson(future)
		future_theta = hpf.transform(sparse.coo_matrix(future.T))
		ds.ca.UMAP_future = umap_embedder.transform(future_theta)

# TODO: evaluate if it was better to use normalized theta than hpf.theta, and what about beta vs hpf.beta?
# TODO: evaluate it it is better to use expected values for S and U vs poisson pooled values
