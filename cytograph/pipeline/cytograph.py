import logging
import warnings

import community
import networkx as nx
import numpy as np
import scipy.sparse as sparse
from umap import UMAP
from numba import NumbaPerformanceWarning, NumbaPendingDeprecationWarning

import loompy
from cytograph.annotation import CellCycleAnnotator
from cytograph.clustering import PolishedLouvain, PolishedSurprise
from cytograph.decomposition import HPF, PCA
from cytograph.embedding import tsne
from cytograph.enrichment import FeatureSelectionByEnrichment, FeatureSelectionByVariance
from cytograph.manifold import BalancedKNN
from cytograph.metrics import jensen_shannon_distance
from cytograph.manifold import PoissonPooling
from cytograph.preprocessing import Normalizer
from cytograph.species import Species

from .config import Config


class Cytograph:
	def __init__(self, *, config: Config) -> None:
		"""
		Run cytograph2

		Args:
			config			The run configuration
		
		Remarks:
			All parameters are obtained from the config object, which comes from the default config
			and can be overridden by the config in the current punchcard
		"""
		self.config = config

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info(f"Running cytograph on {ds.shape[1]} cells")

		species = Species.detect(ds)
		logging.info(f"Species is '{species.name}'")

		logging.info("Recomputing the list of valid genes")
		nnz = ds.map([np.count_nonzero], axis=0)[0]
		valid_genes = np.logical_and(nnz > 10, nnz < ds.shape[1] * 0.6)
		ds.ra.Valid = valid_genes.astype('int')
	
		# Perform Poisson pooling if requested
		main_layer = ""
		if "poisson_pooling" in self.config.steps:
			logging.info(f"Poisson pooling with k_pooling == {self.config.params.k_pooling}")
			main_layer = "pooled"  # if not in config.steps, use the main layer
			pp = PoissonPooling(self.config.params.k_pooling, self.config.params.n_genes, compute_velocity=False, n_threads=self.config.execution.n_cpus, factorization=self.config.params.factorization)
			pp.fit_transform(ds)
		
		# Select features
		if self.config.params.features == "enrichment":
			logging.info(f"Feature selection by enrichment on preliminary clusters")
			if "poisson_pooling" not in self.config.steps:
				logging.info(f"Poisson pooling (for feature selection only) with k_pooling == {self.config.params.k_pooling}")
				pp = PoissonPooling(self.config.params.k_pooling, self.config.params.n_genes, compute_velocity=False, n_threads=self.config.execution.n_cpus, factorization=self.config.params.factorization)
				pp.fit_transform(ds)
			g = nx.from_scipy_sparse_matrix(pp.knn)
			partitions = community.best_partition(g, resolution=1, randomize=False)
			ds.ca.Clusters = np.array([partitions[key] for key in range(pp.knn.shape[0])])
			n_labels = ds.ca.Clusters.max() + 1
			genes = FeatureSelectionByEnrichment(int(self.config.params.n_genes // n_labels), Species.mask(ds, self.config.params.mask), findq=False).select(ds)
		elif self.config.params.features == "variance":
			logging.info(f"Feature selection by variance")
			genes = FeatureSelectionByVariance(self.config.params.n_genes, main_layer, Species.mask(ds, self.config.params.mask)).select(ds)
		logging.info(f"Selected {genes.sum()} genes")

		if self.config.params.factorization in ['PCA', 'PCA+HPF', 'HPF+PCA']:
			logging.info(f"Factorization by PCA")
			normalizer = Normalizer(False)
			normalizer.fit(ds)
			n_components = min(self.config.params.n_factors, ds.shape[1])
			logging.info("PCA projection to %d components", n_components)
			pca = PCA(genes, max_n_components=n_components, layer=main_layer, test_significance=False)
			ds.ca.PCA = pca.fit_transform(ds, normalizer)

		if self.config.params.factorization in ['HPF', 'PCA+HPF', 'HPF+PCA']:
			logging.info(f"Factorization by HPF")
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
		else:
			raise ValueError("params.factorization must be either 'PCA' or 'HPF'")

		if "nn" in self.config.steps or "clustering" in self.config.steps:
			if self.config.params.nn_space in ["PCA", "auto"] and "PCA" in ds.ca:
				transformed = ds.ca.PCA
				metric = "correlation"
			elif self.config.params.nn_space in ["HPF", "auto"] and "HPF" in ds.ca:
				transformed = ds.ca.HPF
				metric = "js"
			else:
				raise ValueError(f"config.params.nn_space = '{self.config.params.nn_space}' is incompatible with config.params.factorization = '{self.config.params.factorization}'")
			logging.info(f"Computing balanced KNN (k = {self.config.params.k}) in {self.config.params.nn_space} space using the '{metric}' metric")
			bnn = BalancedKNN(k=self.config.params.k, metric=metric, maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
			bnn.fit(transformed)
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
			mknn = mknn.tocoo()
			mknn.setdiag(0)
			inside = mknn.data > 1 - radius
			rnn = sparse.coo_matrix((mknn.data[inside], (mknn.row[inside], mknn.col[inside])), shape=mknn.shape)
			ds.col_graphs.RNN = rnn

		if "embeddings" in self.config.steps or "clustering" in self.config.steps:
			logging.info(f"Computing 2D and 3D embeddings from latent space")
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=UserWarning)  # Suppress an annoying UMAP warning about meta-embedding
				warnings.simplefilter("ignore", category=NumbaPerformanceWarning)  # Suppress warnings about numba not being able to parallelize code
				warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)  # Suppress warnings about future deprecations
				perplexity = min(self.config.params.k, (ds.shape[1] - 1) / 3 - 1)
				ds.ca.TSNE = tsne(transformed, metric=metric, perplexity=perplexity)
				ds.ca.UMAP = UMAP(n_components=2, metric=(jensen_shannon_distance if metric == "js" else metric), n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(transformed)
				ds.ca.UMAP3D = UMAP(n_components=3, metric=(jensen_shannon_distance if metric == "js" else metric), n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(transformed)

		if "clustering" in self.config.steps:
			logging.info("Clustering by polished Louvain")
			pl = PolishedLouvain(outliers=False, graph="RNN", embedding="UMAP3D")
			labels = pl.fit_predict(ds)
			ds.ca.ClustersModularity = labels + min(labels)
			ds.ca.OutliersModularity = (labels == -1).astype('int')

			logging.info("Clustering by polished Surprise")
			ps = PolishedSurprise(graph=g, embedding="TSNE")
			labels = ps.fit_predict(ds)
			ds.ca.ClustersSurprise = labels + min(labels)
			ds.ca.OutliersSurprise = (labels == -1).astype('int')

			if self.config.params.clusterer == "louvain":
				ds.ca.Clusters = ds.ca.ClustersModularity
				ds.ca.Outliers = ds.ca.OutliersModularity
			else:
				ds.ca.Clusters = ds.ca.ClustersSurprise
				ds.ca.Outliers = ds.ca.OutliersSurprise

			logging.info(f"Found {ds.ca.Clusters.max() + 1} clusters")

		if species.name in ["Homo sapiens", "Mus musculus"]:
			logging.info(f"Inferring cell cycle")
			CellCycleAnnotator(species).annotate(ds)
