import logging
import warnings
from typing import List

import community
import networkx as nx
import numpy as np
import scipy.sparse as sparse
from scipy.stats import poisson
from umap import UMAP

import loompy
from cytograph.annotation import CellCycleAnnotator
from cytograph.clustering import PolishedLouvain, PolishedSurprise
from cytograph.decomposition import HPF
from cytograph.embedding import tsne
from cytograph.enrichment import FeatureSelectionByEnrichment, FeatureSelectionByVariance
from cytograph.manifold import BalancedKNN
from cytograph.metrics import jensen_shannon_distance
from cytograph.preprocessing import PoissonPooling
from cytograph.species import Species
from cytograph.velocity import VelocityEmbedding, fit_velocity_gamma

from .config import load_config


class Cytograph:
	def __init__(self, *, steps: List[str]) -> None:
		"""
		Run cytograph2

		Args:
			steps							Which steps to include in the analysis
		
		Remarks:
			All parameters are obtained from the config object, which comes from the default config
			and can be overridden by the config in the current punchcard
		"""
		self.steps = steps

	def fit(self, ds: loompy.LoomConnection) -> None:
		config = load_config()

		logging.info(f"Running cytograph on {ds.shape[1]} cells")
		n_samples = ds.shape[1]

		logging.info("Recomputing the list of valid genes")
		nnz = ds.map([np.count_nonzero], axis=0)[0]
		valid_genes = np.logical_and(nnz > 10, nnz < ds.shape[1] * 0.6)
		ds.ra.Valid = valid_genes.astype('int')

		# Perform Poisson pooling if requested, and select features
		if "poisson_pooling" in self.steps:
			logging.info(f"Poisson pooling with k_pooling == {config.params.k_pooling}")
			main_layer = "pooled"
			spliced_layer = "spliced_pooled"
			unspliced_layer = "unspliced_pooled"
			pp = PoissonPooling(config.params.k_pooling, config.params.n_genes, compute_velocity=True)
			pp.fit(ds)
			logging.info(f"Feature selection by enrichment on preliminary clusters")
			g = nx.from_scipy_sparse_matrix(pp.knn)
			partitions = community.best_partition(g, resolution=1, randomize=False)
			ds.ca.Clusters = np.array([partitions[key] for key in range(pp.knn.shape[0])])
			n_labels = ds.ca.Clusters.max() + 1
			genes = FeatureSelectionByEnrichment(int(config.params.n_genes // n_labels), Species.mask(ds, config.params.mask), findq=False).select(ds)
		else:
			logging.info(f"Feature selection by variance")
			main_layer = ""
			spliced_layer = "spliced"
			unspliced_layer = "unspliced"
			genes = FeatureSelectionByVariance(config.params.n_genes, main_layer, Species.mask(ds, config.params.mask)).select(ds)
		
		logging.info(f"Selected {genes.sum()} genes")

		# Load the data for the selected genes
		data = ds[main_layer].sparse(rows=genes).T
		logging.debug(f"Data shape is {data.shape}")

		# HPF factorization
		hpf = HPF(k=config.params.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False, n_threads=config.execution.n_cpus)
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
		# Save the normalized factors
		ds.ra.HPF = beta_all
		ds.ca.HPF = theta

		# HPF factorization of spliced/unspliced
		if "velocity" in self.steps and "spliced" in ds.layers:
			logging.info(f"HPF of spliced molecules")
			data_spliced = ds[spliced_layer].sparse(rows=genes).T
			theta_spliced = hpf.transform(data_spliced)
			ds.ca.HPF_spliced = theta_spliced
			logging.info(f"HPF of unspliced molecules")
			data_unspliced = ds[unspliced_layer].sparse(rows=genes).T
			theta_unspliced = hpf.transform(data_unspliced)
			ds.ca.HPF_unspliced = theta_unspliced

		# Expected values
		logging.info(f"Computing expected values")
		ds["expected"] = 'float32'  # Create a layer of floats
		log_posterior_proba = np.zeros(n_samples)
		theta_unnormalized = hpf.theta
		data = data.toarray()
		start = 0
		batch_size = 6400
		beta_all = ds.ra.HPF_beta  # The unnormalized beta
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
			logging.info(f"Computing balanced KNN (k = {config.params.k}) in HPF latent space")
			bnn = BalancedKNN(k=config.params.k, metric="js", maxl=2 * config.params.k, sight_k=2 * config.params.k, n_jobs=-1)
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
			knn = knn.tocoo()
			knn.setdiag(0)
			inside = knn.data > 1 - radius
			rnn = sparse.coo_matrix((knn.data[inside], (knn.row[inside], knn.col[inside])), shape=knn.shape)
			ds.col_graphs.RNN = rnn

		if "embeddings" in self.steps or "clustering" in self.steps:
			logging.info(f"2D tSNE embedding from latent space")
			ds.ca.TSNE = tsne(theta, metric="js", radius=radius)

			logging.info(f"2D UMAP embedding from latent space")
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=UserWarning)  # Suppress an annoying UMAP warning about meta-embedding
				ds.ca.UMAP = UMAP(n_components=2, metric=jensen_shannon_distance, n_neighbors=config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

			logging.info(f"3D UMAP embedding from latent space")
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=UserWarning)
				ds.ca.UMAP3D = UMAP(n_components=3, metric=jensen_shannon_distance, n_neighbors=config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

		if "clustering" in self.steps:
			logging.info("Clustering by polished Louvain")
			pl = PolishedLouvain(outliers=False)
			labels = pl.fit_predict(ds, graph="RNN", embedding="UMAP3D")
			ds.ca.ClustersModularity = labels + min(labels)
			ds.ca.OutliersModularity = (labels == -1).astype('int')
			if config.params.clusterer == "louvain":
				ds.ca.Clusters = labels + min(labels)
				ds.ca.Outliers = (labels == -1).astype('int')
			logging.info("Clustering by polished Surprise")
			ps = PolishedSurprise(embedding="TSNE")
			labels = ps.fit_predict(ds)
			ds.ca.ClustersSurprise = labels + min(labels)
			ds.ca.OutliersSurprise = (labels == -1).astype('int')
			if config.params.clusterer == "surprise":
				ds.ca.Clusters = labels + min(labels)
				ds.ca.Outliers = (labels == -1).astype('int')
			logging.info(f"Found {ds.ca.Clusters.max() + 1} clusters")

		if "velocity" in self.steps and "spliced" in ds.layers:
			logging.info("Fitting gamma for velocity inference")
			selected = ds.ra.Selected == 1
			n_genes = ds.shape[0]
			s = ds["spliced_exp"][selected, :]
			u = ds["unspliced_exp"][selected, :]
			gamma, _ = fit_velocity_gamma(s, u)
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

		species = Species.detect(ds)
		if species.name in ["Homo sapiens", "Mus musculus"]:
			logging.info("Inferring cell cycle")
			CellCycleAnnotator(species).annotate(ds)
