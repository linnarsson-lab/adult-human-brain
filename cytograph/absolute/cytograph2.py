import cytograph as cg
import numpy as np
import scipy.sparse as sparse
import loompy
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, NearestNeighbors
import logging
from sklearn.manifold import TSNE
from umap import UMAP
from pynndescent import NNDescent
from sklearn.preprocessing import normalize
from typing import *
import os
import community
import networkx as nx
from .velocity_inference import fit_gamma
from .identify_technical_factors import identify_technical_factors
from .metrics import jensen_shannon_distance


class Cytograph2:
	def __init__(self, *, n_genes: int = 2000, n_factors: int = 64, k: int = 50, k_pooling: int = 5, outliers: bool = False, required_genes: str = None, poisson_pooling: bool = True) -> None:
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_pooling = k_pooling
		self.k = k
		self.outliers = outliers
		self.required_genes = required_genes
		self.poisson_pooling = poisson_pooling

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info(f"Running cytograph on {ds.shape[1]} cells")
		n_samples = ds.shape[1]

		# Select genes
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
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
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T  # Normalize so the sums are one because JSD requires it

		if "Batch" in ds.ca and "Replicate" in ds.ca:
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

		# Poisson pooling (in place, except the main layer)
		logging.info(f"Poisson pooling")
		ds["pooled"] = 'int32'
		for (ix, indexes, view) in ds.scan(axis=0):
			if "spliced" in ds.layers:
				ds["spliced"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = ds["spliced"][indexes.min(): indexes.max() + 1, :] + ds["unspliced"][indexes.min(): indexes.max() + 1, :]
			else:
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T

		# Select genes
		logging.info(f"Selecting {self.n_genes} genes after pooling")
		normalizer = cg.Normalizer(False, layer="pooled" if self.poisson_pooling else "")
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes, layer="pooled").fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		# Make sure to include these genes
		genes = np.union1d(genes, np.where(np.isin(ds.ra.Gene, self.required_genes))[0])
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		data = ds["pooled" if self.poisson_pooling else ""].sparse(rows=genes).T

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		beta_all = np.zeros((ds.shape[0], hpf.beta.shape[1]))
		beta_all[genes] = hpf.beta
		# Save the unnormalized factors
		ds.ra.HPF_beta = beta_all
		ds.ca.HPF_theta = hpf.theta
		# Here we normalize so the sums over components are one, because JSD requires it
		# and because otherwise the components will be exactly proportional to cell size
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T
		beta = (hpf.beta.T / hpf.beta.sum(axis=1)).T
		beta_all[genes] = beta
		if "Batch" in ds.ca and "Replicate" in ds.ca:
			technical = identify_technical_factors(theta, ds.ca.Batch, ds.ca.Replicate)
			logging.info(f"Removing {technical.sum()} technical factors")
			theta = theta[:, ~technical]
			beta = beta[:, ~technical]
			beta_all = beta_all[:, ~technical]
		else:
			logging.warn("Could not analyze technical factors because attributes 'Batch' and 'Replicate' are missing")
		# Save the normalized factors
		ds.ra.HPF = beta_all
		ds.ca.HPF = theta

		# HPF factorization of spliced/unspliced
		if "spliced" in ds.layers:
			logging.info(f"HPF of spliced molecules")
			data_spliced = ds["spliced"].sparse(rows=genes).T
			theta_spliced = hpf.transform(data_spliced)
			theta_spliced = (theta_spliced.T / theta_spliced.sum(axis=1)).T
			if "Batch" in ds.ca and "Replicate" in ds.ca:
				theta_spliced = theta_spliced[:, ~technical]
			ds.ca.HPF_spliced = theta_spliced
			logging.info(f"HPF of unspliced molecules")
			data_unspliced = ds["unspliced"].sparse(rows=genes).T
			theta_unspliced = hpf.transform(data_unspliced)
			theta_unspliced = (theta_unspliced.T / theta_unspliced.sum(axis=1)).T
			if "Batch" in ds.ca and "Replicate" in ds.ca:
				theta_unspliced = theta_unspliced[:, ~technical]
			ds.ca.HPF_unspliced = theta_unspliced

		# Expected values
		logging.info(f"Computing expected values")
		ds["expected"] = 'float32'  # Create a layer of floats
		start = 0
		batch_size = 6400
		if "spliced" in ds.layers:
			ds["spliced_exp"] = 'float32'
			ds['unspliced_exp'] = 'float32'
		while start < n_samples:
			# Compute PPV
			ds["expected"][:, start: start + batch_size] = beta_all @ theta[start: start + batch_size, :].T
			if "spliced" in ds.layers:
				ds["spliced_exp"][:, start: start + batch_size] = beta_all @ theta_spliced[start: start + batch_size, :].T
				ds["unspliced_exp"][:, start: start + batch_size] = beta_all @ theta_unspliced[start: start + batch_size, :].T
			start += batch_size

		logging.info(f"Computing balanced KNN (k = {self.k}) in latent space")
		bnn = cg.BalancedKNN(k=self.k, metric="js", maxl=2 * self.k, sight_k=2 * self.k, n_jobs=-1)
		bnn.fit(theta.astype("float64"))  # Not sure why, but with float32 BalancedKNN throws *** Error in `/home/sten/anaconda3/bin/python': double free or corruption (out): 0x00005648e27189c0 ***
		knn = bnn.kneighbors_graph(mode='distance')
		mknn = knn.minimum(knn.transpose())
		# Convert distances to similarities
		knn.data = 1 - knn.data
		mknn.data = 1 - mknn.data
		ds.col_graphs.KNN = knn
		ds.col_graphs.MKNN = mknn
		# Compute the effective resolution
		d = 1 - knn.data
		d = d[d < 1]
		radius = np.percentile(d, 90)
		logging.info(f"Found effective radius {radius:.02}")
		ds.attrs.radius = radius
		knn.setdiag(0)
		knn = knn.tocoo()
		inside = knn.data > 1 - radius
		rnn = sparse.coo_matrix((knn.data[inside], (knn.row[inside], knn.col[inside])), shape=knn.shape)
		ds.col_graphs.RNN = rnn

		logging.info(f"Computing balanced KNN (k = {10 * self.k}) in latent space")
		# This stage computes an RNN with ten times as many neighbors, but still using the same radius
		# This will expand the neighborhoods in regions of high density, without causing it to bleed outside regions of low density
		bnn = cg.BalancedKNN(k=10 * self.k, metric="js", maxl=2 * self.k, sight_k=20 * self.k, n_jobs=-1)
		bnn.fit(theta.astype("float64"))
		knn = bnn.kneighbors_graph(mode='distance')
		logging.info("3")
		# Convert distances to similarities
		knn.data = 1 - knn.data
		logging.info("4")
		knn.setdiag(0)
		logging.info("5")
		knn = knn.tocoo()
		logging.info("6")
		inside = knn.data > 1 - radius
		logging.info("7")
		rnn = sparse.coo_matrix((knn.data[inside], (knn.row[inside], knn.col[inside])), shape=knn.shape)
		logging.info("8")
		ds.col_graphs.RNN10X = rnn
		logging.info("9")

		logging.info(f"2D tSNE embedding from latent space")
		tsne = cg.tsne_js(theta, radius=radius)
		ds.ca.TSNE = tsne

		logging.info(f"2D UMAP embedding from latent space")
		umap_embedder = UMAP(metric=jensen_shannon_distance, spread=2, repulsion_strength=2, n_neighbors=self.k, n_components=2)
		umap = umap_embedder.fit_transform(theta)
		ds.ca.UMAP = umap

		logging.info(f"3D UMAP embedding from latent space")
		umap3d = UMAP(metric=jensen_shannon_distance, spread=2, repulsion_strength=2, n_neighbors=self.k, n_components=3).fit_transform(theta)
		ds.ca.UMAP3D = umap3d

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain(outliers=self.outliers)
		labels = pl.fit_predict(ds, graph="RNN", embedding="HPF")
		ds.ca.Clusters = labels + min(labels)
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {labels.max() + 1} clusters")

		if "spliced" in ds.layers:
			logging.info("Fitting gamma for velocity inference")
			selected = ds.ra.Selected == 1
			n_genes = ds.shape[0]
			s = ds["spliced_exp"][selected, :]
			u = ds["unspliced_exp"][selected, :]
			gamma, _ = fit_gamma(s, u)
			gamma_all = np.zeros(n_genes)
			gamma_all[selected] = gamma
			ds.ra.Gamma = gamma_all

			logging.info("Computing velocity")
			velocity = u - gamma[:, None] * s
			ds["velocity"] = "float32"
			ds["velocity"][selected, :] = velocity
