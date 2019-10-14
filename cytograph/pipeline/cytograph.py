import logging
import warnings

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
from cytograph.cytograph1 import PCAProjection, Normalizer, TSNE

from .config import Config


class Cytograph:
	def __init__(self, *, config: Config) -> None:
		"""
		Run cytograph2

		Args:
			steps							Which steps to include in the analysis
		
		Remarks:
			All parameters are obtained from the config object, which comes from the default config
			and can be overridden by the config in the current punchcard
		"""
		self.config = config

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info(f"Running cytograph on {ds.shape[1]} cells")
		n_samples = ds.shape[1]

		species = Species.detect(ds)
		logging.info(f"Species is '{species.name}'")

		logging.info("Recomputing the list of valid genes")
		nnz = ds.map([np.count_nonzero], axis=0)[0]
		valid_genes = np.logical_and(nnz > 10, nnz < ds.shape[1] * 0.6)
		ds.ra.Valid = valid_genes.astype('int')
	
		# Perform Poisson pooling if requested
		if "poisson_pooling" in self.config.steps:
			logging.info(f"Poisson pooling with k_pooling == {self.config.params.k_pooling}")
			main_layer = "pooled"
			spliced_layer = "spliced_pooled"
			unspliced_layer = "unspliced_pooled"
			pp = PoissonPooling(self.config.params.k_pooling, self.config.params.n_genes, compute_velocity=True, n_threads=self.config.execution.n_cpus)
			pp.fit_transform(ds)
		else:
			main_layer = ""
			spliced_layer = "spliced"
			unspliced_layer = "unspliced"
		
		# Select features
		if self.config.params.features == "enrichment":
			logging.info(f"Feature selection by enrichment on preliminary clusters")
			if "poisson_pooling" not in self.config.steps:
				pp = PoissonPooling(self.config.params.k_pooling, self.config.params.n_genes, compute_velocity=True, n_threads=self.config.execution.n_cpus)
				pp.fit(ds)
			g = nx.from_scipy_sparse_matrix(pp.knn)
			partitions = community.best_partition(g, resolution=1, randomize=False)
			ds.ca.Clusters = np.array([partitions[key] for key in range(pp.knn.shape[0])])
			n_labels = ds.ca.Clusters.max() + 1
			genes = FeatureSelectionByEnrichment(int(self.config.params.n_genes // n_labels), Species.mask(ds, self.config.params.mask), findq=False).select(ds)
		elif self.config.params.features == "variance":
			logging.info(f"Feature selection by variance")
			genes = FeatureSelectionByVariance(self.config.params.n_genes, main_layer, Species.mask(ds, self.config.params.mask)).select(ds)
		logging.info(f"Selected {genes.sum()} genes")

		if self.config.params.factorization == 'PCA':
			logging.info("Normalization")
			normalizer = Normalizer(False)
			normalizer.fit(ds)
			n_components = min(50, ds.shape[1])
			logging.info("PCA projection to %d components", n_components)
			pca = PCAProjection(genes, max_n_components=n_components)
			transformed = pca.fit_transform(ds, normalizer)
			ds.ca.PCA = transformed
		elif self.config.params.factorization == 'HPF':
			# Load the data for the selected genes
			data = ds[main_layer].sparse(rows=genes).T
			logging.debug(f"Data shape is {data.shape}")

			# HPF factorization
			hpf = HPF(k=self.config.params.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False, n_threads=self.config.execution.n_cpus)
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
			if "velocity" in self.config.steps and "spliced" in ds.layers:
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
			if "velocity" in self.config.steps and "spliced" in ds.layers:
				ds["spliced_exp"] = 'float32'
				ds['unspliced_exp'] = 'float32'
			while start < n_samples:
				# Compute PPV (using normalized theta)
				ds["expected"][:, start: start + batch_size] = beta_all @ theta[start: start + batch_size, :].T
				# Compute PPV using raw theta, for calculating posterior probability of the observations
				ppv_unnormalized = beta @ theta_unnormalized[start: start + batch_size, :].T
				log_posterior_proba[start: start + batch_size] = poisson.logpmf(data.T[:, start: start + batch_size], ppv_unnormalized).sum(axis=0)
				if "velocity" in self.config.steps and "spliced" in ds.layers:
					ds["spliced_exp"][:, start: start + batch_size] = beta_all @ theta_spliced[start: start + batch_size, :].T
					ds["unspliced_exp"][:, start: start + batch_size] = beta_all @ theta_unspliced[start: start + batch_size, :].T
				start += batch_size
			ds.ca.HPF_LogPP = log_posterior_proba

		if "nn" in self.config.steps or "clustering" in self.config.steps:
			if self.config.params.factorization == 'PCA':
				logging.info(f"Computing balanced KNN (k = {self.config.params.k}) in PCA space")
				bnn = BalancedKNN(k=self.config.params.k, metric="euclidean", maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
				bnn.fit(transformed)
				knn = bnn.kneighbors_graph(mode='distance')
				knn.eliminate_zeros()
				mknn = knn.minimum(knn.transpose())
				ds.col_graphs.KNN = knn
				ds.col_graphs.MKNN = mknn
			elif self.config.params.factorization == 'HPF':
				logging.info(f"Computing balanced KNN (k = {self.config.params.k}) in HPF latent space")
				bnn = BalancedKNN(k=self.config.params.k, metric="js", maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
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

		if "embeddings" in self.config.steps or "clustering" in self.config.steps:
			logging.info(f"2D tSNE embedding from latent space")
			if self.config.params.factorization == 'PCA':
				perplexity = min(self.config.params.k, (ds.shape[1] - 1) / 3 - 1)
				ds.ca.TSNE = TSNE(perplexity=perplexity).layout(transformed, knn=knn.tocsr())
				ds.ca.UMAP = UMAP(n_components=2, n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(transformed)
				ds.ca.UMAP3D = UMAP(n_components=3, n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(transformed)
			else:
				ds.ca.TSNE = tsne(theta, metric="js", radius=radius)

				logging.info(f"2D UMAP embedding from latent space")
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=UserWarning)  # Suppress an annoying UMAP warning about meta-embedding
					ds.ca.UMAP = UMAP(n_components=2, metric=jensen_shannon_distance, n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

				logging.info(f"3D UMAP embedding from latent space")
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=UserWarning)
					ds.ca.UMAP3D = UMAP(n_components=3, metric=jensen_shannon_distance, n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

		if "clustering" in self.config.steps:
			logging.info("Clustering by polished Louvain")
			pl = PolishedLouvain(outliers=False)
			if self.config.params.factorization == 'PCA':
				labels = pl.fit_predict(ds, graph="MKNN", embedding="UMAP3D")
			else:
				labels = pl.fit_predict(ds, graph="RNN", embedding="UMAP3D")
			ds.ca.ClustersModularity = labels + min(labels)
			ds.ca.OutliersModularity = (labels == -1).astype('int')
			if self.config.params.clusterer == "louvain":
				ds.ca.Clusters = labels + min(labels)
				ds.ca.Outliers = (labels == -1).astype('int')
			if self.config.params.factorization != 'PCA':
				logging.info("Clustering by polished Surprise")
				ps = PolishedSurprise(embedding="TSNE")
				labels = ps.fit_predict(ds)
				ds.ca.ClustersSurprise = labels + min(labels)
				ds.ca.OutliersSurprise = (labels == -1).astype('int')
				if self.config.params.clusterer == "surprise":
					ds.ca.Clusters = labels + min(labels)
					ds.ca.Outliers = (labels == -1).astype('int')
			logging.info(f"Found {ds.ca.Clusters.max() + 1} clusters")

		if "velocity" in self.config.steps and "spliced" in ds.layers:
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

			if "embeddings" in self.config.steps:
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

		if species.name in ["Homo sapiens", "Mus musculus"]:
			logging.info(f"Inferring cell cycle")
			CellCycleAnnotator(species).annotate(ds)
