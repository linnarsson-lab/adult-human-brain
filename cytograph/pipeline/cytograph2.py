import cytograph as cg
import numpy as np
import scipy.sparse as sparse
from scipy.interpolate import griddata
from scipy.stats import poisson
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
from .velocity_embedding import VelocityEmbedding
from .neighborhood_enrichment import NeighborhoodEnrichment
from .embedding import tsne
from .species import Species


class Cytograph2:
	def __init__(self, *, n_genes: int = 2000, n_factors: int = 64, k: int = 50, k_pooling: int = 5, outliers: bool = False, required_genes: List[str] = None, mask_cell_cycle: bool = False, feature_selection_method: str = "variance", steps: List[str]) -> None:
		"""
		Run cytograph2

		Args:
			n_genes							Number of genes to select
			n_factors						Number of HPF factors
			k								Number of neighbors for KNN graph
			k_pooling						Number of neighbors for Poisson pooling
			outliers						Allow outliers and mark them
			required_genes					List of genes that must be included in any feature selection (except "cellcycle")
			mask_cell_cycle					Remove cell cycle genes (including from required_genes), unless feature_selection_method == "cellcycle"
			feature_selection_method 		"markers", "variance" or "cellcycle"
			steps							Which steps to include in the analysis
		"""
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_pooling = k_pooling
		self.k = k
		self.outliers = outliers
		self.required_genes = required_genes
		self.mask_cell_cycle = mask_cell_cycle
		self.feature_selection_method = feature_selection_method
		self.steps = steps

	def feature_selection_by_cell_cycle(self, ds: loompy.LoomConnection, main_layer: str) -> np.ndarray:
		cc_genes = Species(ds).cell_cycle_genes
		genes = np.where(np.isin(ds.ra.Gene, cc_genes))[0]
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		return genes

	def feature_selection_by_variance(self, ds: loompy.LoomConnection, main_layer: str) -> np.ndarray:
		cc_genes = Species(ds).cell_cycle_genes
		normalizer = cg.Normalizer(False, layer=main_layer)
		normalizer.fit(ds)
		mask = None
		if self.mask_cell_cycle:
			mask = np.isin(ds.ra.Gene, cc_genes)
		genes = cg.FeatureSelection(self.n_genes, layer=main_layer).fit(ds, mu=normalizer.mu, sd=normalizer.sd, mask=mask)
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		return genes

	def feature_selection_by_markers(self, ds: loompy.LoomConnection, main_layer: str) -> np.ndarray:
		cc_genes = Species(ds).cell_cycle_genes

		logging.info("Selecting up to %d marker genes", self.n_genes)
		normalizer = cg.Normalizer(False, layer=main_layer)
		normalizer.fit(ds)
		mask = None
		if self.mask_cell_cycle:
			mask = np.isin(ds.ra.Gene, cc_genes)
		genes = cg.FeatureSelection(self.n_genes, layer=main_layer).fit(ds, mu=normalizer.mu, sd=normalizer.sd, mask=mask)
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
		(genes, _, _) = cg.MarkerSelection(n_markers=int(self.n_genes / n_labels), findq=False, mask=mask).fit(ds)
		if temp is not None:
			ds.ca.Clusters = temp

		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		return genes

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info(f"Running cytograph on {ds.shape[1]} cells")
		n_samples = ds.shape[1]

		if "poisson_pooling" in self.steps and ("pooled" in ds.layers):
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
		elif self.feature_selection_method == "cellcycle":
			genes = self.feature_selection_by_cell_cycle(ds, main_layer)

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
		if "batch_correction" in self.steps and "Batch" in ds.ca and "Replicate" in ds.ca:
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
			if "batch_correction" in self.steps and "Batch" in ds.ca and "Replicate" in ds.ca:
				theta_spliced = theta_spliced[:, ~technical]
			ds.ca.HPF_spliced = theta_spliced
			logging.info(f"HPF of unspliced molecules")
			data_unspliced = ds[unspliced_layer].sparse(rows=genes).T
			theta_unspliced = hpf.transform(data_unspliced)
			theta_unspliced = (theta_unspliced.T / theta_unspliced.sum(axis=1)).T
			if "batch_correction" in self.steps and "Batch" in ds.ca and "Replicate" in ds.ca:
				theta_unspliced = theta_unspliced[:, ~technical]
			ds.ca.HPF_unspliced = theta_unspliced

		# Expected values
		logging.info(f"Computing expected values")
		ds["expected"] = 'float32'  # Create a layer of floats
		log_posterior_proba = np.zeros(n_samples)
		theta_unnormalized = hpf.theta[:, ~technical] if "batch_correction" in self.steps else hpf.theta
		data = data.toarray()
		start = 0
		batch_size = 6400
		if "velocity" in self.steps and "spliced" in ds.layers:
			ds["spliced_exp"] = 'float32'
			ds['unspliced_exp'] = 'float32'
		while start < n_samples:
			# Compute PPV (using normalized theta)
			ds["expected"][:, start: start + batch_size] = beta_all @ theta[start: start + batch_size, :].T
			# Compute PPV using raw theta, for calculating posterior probability of the observations
			ppv_unnormalized = beta @ theta_unnormalized[start: start + batch_size, :].T
			log_posterior_proba[start: start + batch_size] = poisson.logpmf(data.T[:, start: start + batch_size], ppv_unnormalized).sum(axis=0)
			if "velocity" in self.steps and "spliced" in ds.layers:
				ds["spliced_exp"][:, start: start + batch_size] = beta_all @ theta_spliced[start: start + batch_size, :].T
				ds["unspliced_exp"][:, start: start + batch_size] = beta_all @ theta_unspliced[start: start + batch_size, :].T
			start += batch_size
		ds.ca.HPF_LogPP = log_posterior_proba

		if "nn" in self.steps or "clustering" in self.steps:
			logging.info(f"Computing balanced KNN (k = {self.k}) in latent space")
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

		if "embeddings" in self.steps or "clustering" in self.steps:
			logging.info(f"2D tSNE embedding from latent space")
			ds.ca.TSNE = tsne(theta, metric="js", radius=radius)

			logging.info(f"2D UMAP embedding from latent space")
			ds.ca.UMAP = UMAP(n_components=2, metric=jensen_shannon_distance, n_neighbors=self.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

			logging.info(f"3D UMAP embedding from latent space")
			ds.ca.UMAP3D = UMAP(n_components=3, metric=jensen_shannon_distance, n_neighbors=self.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

		if "clustering" in self.steps:
			logging.info("Clustering by polished Louvain")
			pl = cg.PolishedLouvain(outliers=self.outliers)
			labels = pl.fit_predict(ds, graph="RNN", embedding="UMAP3D")
			ds.ca.Clusters = labels + min(labels)
			ds.ca.Outliers = (labels == -1).astype('int')
			logging.info(f"Found {labels.max() + 1} clusters")

		if "velocity" in self.steps and "spliced" in ds.layers:
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
			ds.ca.HPFVelocity = np.dot(np.linalg.pinv(beta[ds.ra.Selected == 1]), velocity).T

			if "embeddings" in self.steps:
				logging.info("Projecting velocity to TSNE 2D embedding")
				ve = VelocityEmbedding(data_source="HPF", velocity_source="HPFVelocity", embedding_name="TSNE", neighborhood_type="RNN", points_kind="cells", min_neighbors=0)
				ds.ca.TSNEVelocity = ve.fit(ds)
				# Embed velocity on a 50x50 grid
				ve = VelocityEmbedding(data_source="HPF", velocity_source="HPFVelocity", embedding_name="TSNE", neighborhood_type="radius", neighborhood_size=5, points_kind="grid", num_points=50, min_neighbors=5)
				ds.attrs.TSNEVelocity = ve.fit(ds)
				ds.attrs.TSNEVelocityPoints = ve.points

				logging.info("Projecting velocity to UMAP 2D embedding")
				ve = VelocityEmbedding(data_source="HPF", velocity_source="HPFVelocity", embedding_name="UMAP", neighborhood_type="RNN", points_kind="cells", min_neighbors=0)
				ds.ca.UMAPVelocity = ve.fit(ds)
				# Embed velocity on a 50x50 grid
				ve = VelocityEmbedding(data_source="HPF", velocity_source="HPFVelocity", embedding_name="UMAP", neighborhood_type="radius", neighborhood_size=0.5, points_kind="grid", num_points=50, min_neighbors=5)
				ds.attrs.UMAPVelocity = ve.fit(ds)
				ds.attrs.UMAPVelocityPoints = ve.points

		if Species(ds).name in ["Homo sapiens", "Mus musculus"]:
			logging.info("Inferring cell cycle")
			cca = CellCycleAnnotator(ds)
			cca.annotate_loom()

