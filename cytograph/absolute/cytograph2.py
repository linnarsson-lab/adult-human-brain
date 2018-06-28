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
import igraph


class Cytograph2:
	def __init__(self, accel: bool = False, log: bool = True, normalize: bool = True, n_genes: int = 1000, n_factors: int = 100, a: float = 1, b: float = 10, c: float = 1, d: float = 10, k: int = 10, k_smoothing: int = 100, max_iter: int = 200) -> None:
		self.accel = accel
		self.log = log
		self.normalize = normalize
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_smoothing = k_smoothing
		self.k = k
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.max_iter = max_iter

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
		if self.accel:
			hpf = cg.HPF_accel(a=self.a, b=self.b, c=self.c, d=self.d, k=self.n_factors, max_iter=self.max_iter)
		else:
			hpf = cg.HPF(a=self.a, b=self.b, c=self.c, d=self.d, k=self.n_factors, max_iter=self.max_iter)
		hpf.fit(data)

		# KNN in HPF space
		logging.info(f"Computing KNN (k={self.k_smoothing}) in latent space")
		if self.log:
			theta = np.log(hpf.theta)
		else:
			theta = hpf.theta
		if self.normalize:
			theta = (theta - theta.min(axis=0))
			theta = theta / theta.max(axis=0)
		hpfn = normalize(theta)  # This converts euclidean distances to cosine distances (ball_tree doesn't directly support cosine)
		nn = NearestNeighbors(self.k_smoothing, algorithm="ball_tree", metric='euclidean', n_jobs=4)
		nn.fit(hpfn)
		knn = nn.kneighbors_graph(hpfn, mode='connectivity')
		knn.setdiag(1)

		# Poisson smoothing (in place)
		logging.info(f"Poisson smoothing")
		ds["smoothened"] = 'int32'
		if "spliced" in ds.layers:
			ds["spliced_ps"] = 'int32'
			ds["unspliced_ps"] = 'int32'
		for (ix, indexes, view) in ds.scan(axis=0):
			if "spliced" in ds.layers:
				ds["spliced_ps"][indexes.min(): indexes.max() + 1, :] = knn.dot(view["spliced"][:, :].T).T
				ds["unspliced_ps"][indexes.min(): indexes.max() + 1, :] = knn.dot(view["unspliced"][:, :].T).T
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
		if self.accel:
			hpf = cg.HPF_accel(a=self.a, b=self.b, c=self.c, d=self.d, k=self.n_factors, max_iter=self.max_iter)
		else:
			hpf = cg.HPF(a=self.a, b=self.b, c=self.c, d=self.d, k=self.n_factors, max_iter=self.max_iter)
		hpf.fit(data)

		logging.info(f"Saving normalized latent factors")
		if self.log:
			beta = np.log(hpf.beta)
		else:
			beta = hpf.beta
		if self.normalize:
			beta = (beta - beta.min(axis=0))
			beta = beta / beta.max(axis=0)
		beta_all = np.zeros((ds.shape[0], beta.shape[1]))
		beta_all[genes] = beta
		ds.ra.HPF = beta_all

		if self.log:
			theta = np.log(hpf.theta)
		else:
			theta = hpf.theta
		if normalize:
			theta = (theta - theta.min(axis=0))
			theta = theta / theta.max(axis=0)
		totals = ds.map([np.sum], axis=1)[0]
		theta = (theta.T / totals).T * np.median(totals)
		ds.ca.HPF = theta

		logging.info(f"tSNE embedding from latent space")
		tsne = TSNE(metric="cosine").fit_transform(theta)
		ds.ca.TSNE = tsne

		logging.info(f"Computing balanced KNN (k = {self.k}) in latent space")
		hpfn = normalize(theta)  # This makes euclidean distances equivalent to cosine distances (ball_tree doesn't support cosine)
		bnn = cg.BalancedKNN(k=self.k, metric="euclidean", maxl=2 * self.k, sight_k=2 * self.k)
		bnn.fit(hpfn)
		knn = bnn.kneighbors_graph(mode='connectivity')
		mknn = knn.minimum(knn.transpose())
		ds.col_graphs.KNN = knn
		ds.col_graphs.MKNN = mknn

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain()
		labels = pl.fit_predict(ds, "KNN")
		ds.ca.Clusters = labels + 1
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {labels.max() + 1} clusters")
