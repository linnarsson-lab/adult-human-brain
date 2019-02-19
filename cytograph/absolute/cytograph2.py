import cytograph as cg
import numpy as np
import scipy.sparse as sparse
from scipy.interpolate import griddata
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
from .cell_cycle_annotator import CellCycleAnnotator


class Cytograph2:
	def __init__(self, *, n_genes: int = 2000, n_factors: int = 64, k: int = 50, k_pooling: int = 5, outliers: bool = False, required_genes: str = None, feature_selection_method: str = "markers", use_poisson_pooling: bool = True) -> None:
		"""
		Run cytograph2

		Args:
			n_genes							Number of genes to select
			n_factors						Number of HPF factors
			k								Number of neighbors for KNN graph
			k_pooling						Number of neighbors for Poisson pooling
			outliers						Allow outliers and mark them
			required_genes					List of genes that must be included in any feature selection
			feature_selection_method 		"markers" or "variance"
			use_poisson_pooling				If true and pooling layers exist, use them
		"""
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_pooling = k_pooling
		self.k = k
		self.outliers = outliers
		self.required_genes = required_genes
		self.feature_selection_method = feature_selection_method
		self.use_poisson_pooling = use_poisson_pooling

	def embed_velocity(self, ds: loompy.LoomConnection, n_bins: int = 50, gradient_spacing: int = 2, embedding: str = "TSNE") -> None:
		"""
		Project velocities from latent (HPF) space to a grid overlaid on a 2D embedding (e.g. TSNE or UMAP)

		Args:
			ds					LoomConnection
			n_bins				Number of grid points (will be used for both x and y)
			gradient_spacing	Spacing to use when computing gradient of theta in the embedding
			embedding			Name of the embedding (e.g. "TSNE" or "UMAP")
		
		Returns: 				(nothing)

		Remarks:
			This method will estimate a local velocity at each point on a grid on the embedding. For each grid square,
			the local velocity is computed by solving the least-squares equation

				velocity_hpf = theta_gradient @ velocity_embedding
			
			where "velocity_hpf" is the average velocity in HPF latent space for cells in the square, "theta_gradient"
			is the gradient of theta (averaged in the square) across the embedding, and "velocity_embedding" is the
			unknown velocity vector in the embedding.

			Only 2D embeddings are supported currently.
		"""
		theta = ds.ca.HPF
		n_components = theta.shape[1]
		velocity_hpf = ds.ca.VelocityHPF
		points = ds.ca[embedding]
		xmin, xmax = points[:, 0].min(), points[:, 0].max()
		ymin, ymax = points[:, 1].min(), points[:, 1].max()
		grid_x, grid_y = np.mgrid[xmin:xmax:complex(0, n_bins), ymin:ymax:complex(0, n_bins)]  # type: ignore

		# Interpolate velocity and theta on a regular grid
		theta_binned = np.zeros((n_bins, n_bins, n_components))
		velocity_binned = np.zeros((n_bins, n_bins, n_components))
		for c in range(n_components):
			theta_binned[:, :, c] = np.nan_to_num(griddata(points, theta[:, c], (grid_x, grid_y), method='linear'), copy=False)
			velocity_binned[:, :, c] = np.nan_to_num(griddata(points, velocity_hpf[:, c], (grid_x, grid_y), method='linear'), copy=False)

		# take the gradient of theta w.r.t. x and y in the embedding, independently for each component
		theta_grad = np.stack(np.gradient(theta_binned, gradient_spacing, axis=(0, 1)), axis=3)  # (n_bins, n_bins, n_components, 2)
		# Use least squares to estimate velocity on the embedding
		arrows = np.zeros((n_bins, n_bins, 2))
		for i in range(n_bins):
			for j in range(n_bins):
				arrows[i,j,:], _, _, _ = np.linalg.lstsq(theta_grad[i,j], velocity_binned[i,j,:],rcond=None)
		ds.attrs["EmbeddedVelocity" + embedding] = arrows
	
	def poisson_pooling(self, ds: loompy.LoomConnection) -> None:
		n_samples = ds.shape[1]
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

		# Poisson pooling
		logging.info(f"Poisson pooling")
		ds["pooled"] = 'int32'
		if "spliced" in ds.layers:
			ds["spliced_pooled"] = 'int32'
			ds["unspliced_pooled"] = 'int32'
		for (ix, indexes, view) in ds.scan(axis=0):
			if "spliced" in ds.layers:
				ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] + ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :]
			else:
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T

	def feature_selection_by_variance(self, ds: loompy.LoomConnection, main_layer: str) -> np.ndarray:
		normalizer = cg.Normalizer(False, layer=main_layer)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes, layer=main_layer).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		# Make sure to include these genes
		genes = np.union1d(genes, np.where(np.isin(ds.ra.Gene, self.required_genes))[0])
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		return genes

	def feature_selection_by_markers(self, ds: loompy.LoomConnection, main_layer: str) -> np.ndarray:
		logging.info("Selecting up to %d marker genes", self.n_genes)
		normalizer = cg.Normalizer(False, layer=main_layer)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes, layer=main_layer).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		n_cells = ds.shape[1]
		n_components = min(50, n_cells)
		logging.info("PCA projection to %d components", n_components)
		pca = cg.PCAProjection(genes, max_n_components=n_components, layer=main_layer)
		pca_transformed = pca.fit_transform(ds, normalizer)
		transformed = pca_transformed

		logging.info("Generating balanced KNN graph")
		np.random.seed(0)
		k = min(self.k, n_cells - 1)
		bnn = cg.BalancedKNN(k=k, maxl=2 * k, sight_k=2 * k)
		bnn.fit(transformed)
		knn = bnn.kneighbors_graph(mode='connectivity')
		knn = knn.tocoo()
		mknn = knn.minimum(knn.transpose()).tocoo()

		logging.info("MKNN-Louvain clustering with outliers")
		(a, b, w) = (mknn.row, mknn.col, mknn.data)
		lj = cg.LouvainJaccard(resolution=1, jaccard=False)
		labels = lj.fit_predict(knn)
		bigs = np.where(np.bincount(labels) >= 10)[0]
		mapping = {k: v for v, k in enumerate(bigs)}
		labels = np.array([mapping[x] if x in bigs else -1 for x in labels])

		n_labels = np.max(labels) + 1
		logging.info("Found " + str(n_labels) + " preliminary clusters")

		logging.info("Marker selection")
		temp = None
		if "Clusters" in ds.ca:
			temp = ds.ca.Clusters
		ds.ca.Clusters = labels - labels.min()
		(genes, _, _) = cg.MarkerSelection(n_markers=int(500 / n_labels), findq=False).fit(ds)
		if temp is not None:
			ds.ca.Clusters = temp
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		return genes

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info(f"Running cytograph on {ds.shape[1]} cells")
		n_samples = ds.shape[1]

		if self.use_poisson_pooling and ("pooled" in ds.layers):
			main_layer = "pooled"
			spliced_layer = "spliced_pooled"
			unspliced_layer = "unspliced_pooled"
		else:
			main_layer = ""
			spliced_layer = "spliced"
			unspliced_layer = "unspliced"
		# Select genes
		logging.info(f"Selecting {self.n_genes} genes")
		if self.feature_selection_method == "variance":
			genes = self.feature_selection_by_variance(ds, main_layer)
		elif self.feature_selection_method == "markers":
			genes = self.feature_selection_by_markers(ds, main_layer)

		# Load the data for the selected genes
		data = ds[main_layer].sparse(rows=genes).T

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
			data_spliced = ds[spliced_layer].sparse(rows=genes).T
			theta_spliced = hpf.transform(data_spliced)
			theta_spliced = (theta_spliced.T / theta_spliced.sum(axis=1)).T
			if "Batch" in ds.ca and "Replicate" in ds.ca:
				theta_spliced = theta_spliced[:, ~technical]
			ds.ca.HPF_spliced = theta_spliced
			logging.info(f"HPF of unspliced molecules")
			data_unspliced = ds[unspliced_layer].sparse(rows=genes).T
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

		# logging.info(f"Computing balanced KNN (k = {self.k}) in latent space")
		bnn = cg.BalancedKNN(k=self.k, metric="js", maxl=2 * self.k, sight_k=2 * self.k, n_jobs=-1)
		bnn.fit(theta)
		knn = bnn.kneighbors_graph(mode='distance')
		knn.eliminate_zeros()
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
		logging.info(f"90th percentile radius: {radius:.02}")
		ds.attrs.radius = radius
		knn.setdiag(0)
		knn = knn.tocoo()
		inside = knn.data > 1 - radius
		rnn = sparse.coo_matrix((knn.data[inside], (knn.row[inside], knn.col[inside])), shape=knn.shape)
		ds.col_graphs.RNN = rnn

		logging.info(f"2D tSNE embedding from latent space")
		tsne = cg.tsne_js(theta, radius=radius)
		ds.ca.TSNE = tsne

		logging.info(f"2D UMAP embedding from latent space")
		umap2d = UMAP(n_components=2, metric=jensen_shannon_distance, n_neighbors=self.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)
		ds.ca.UMAP = umap2d

		logging.info(f"3D UMAP embedding from latent space")
		umap3d = UMAP(n_components=3, metric=jensen_shannon_distance, n_neighbors=self.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)
		ds.ca.UMAP3D = umap3d

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain(outliers=self.outliers)
		labels = pl.fit_predict(ds, graph="RNN", embedding="UMAP3D")
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

			logging.info("Projecting velocity to latent space")
			beta = ds.ra.HPF
			ds.ca.VelocityHPF = (beta[ds.ra.Selected == 1].T @ velocity).T

			logging.info("Projecting velocity to TSNE 2D embedding")
			self.embed_velocity(ds, embedding="TSNE")

			logging.info("Projecting velocity to UMAP 2D embedding")
			self.embed_velocity(ds, embedding="UMAP")

		logging.info("Inferring cell cycle")
		CellCycleAnnotator(ds).annotate_loom()
