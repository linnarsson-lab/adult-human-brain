import logging
import warnings

import community
import networkx as nx
import numpy as np
import scipy.sparse as sparse
from umap import UMAP
from sklearn.manifold import TSNE
from numba import NumbaPerformanceWarning, NumbaPendingDeprecationWarning
from scipy.sparse import SparseEfficiencyWarning
from pynndescent import NNDescent
from sknetwork.clustering import Louvain

import loompy
from cytograph.annotation import CellCycleAnnotator
from cytograph.clustering import PolishedLouvain, PolishedSurprise
from cytograph.decomposition import HPF, PCA
from cytograph.embedding import art_of_tsne
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
		if self.config.params.factorization not in ["PCA", "HPF", "both"]:
			raise ValueError("params.factorization must be either 'PCA' or 'HPF' or 'both'")
		if self.config.params.features not in ["enrichment", "variance"]:
			raise ValueError("params.features must be either 'enrichment' or 'variance'")
		if self.config.params.nn_space not in ["PCA", "HPF", "auto"]:
			raise ValueError("params.nn_space must be either 'PCA' or 'HPF' or 'auto'")
		if not ((self.config.params.nn_space in ["PCA", "auto"] and self.config.params.factorization in ["PCA", "both"]) or (self.config.params.nn_space in ["HPF", "auto"] and self.config.params.factorization in ["HPF", "both"])):
			raise ValueError(f"config.params.nn_space = '{self.config.params.nn_space}' is incompatible with config.params.factorization = '{self.config.params.factorization}'")

		species = Species.detect(ds)
		logging.info(f"Species is '{species.name}'")

		logging.info("Recomputing the list of valid genes")
		nnz = ds.map([np.count_nonzero], axis=0)[0]
		valid_genes = (nnz > 10) & (nnz < ds.shape[1] * 0.6)
		ds.ra.Valid = valid_genes.astype('int')

		# Perform Poisson pooling if requested
		main_layer = ""
		if "poisson_pooling" in self.config.steps:
			logging.info(f"Poisson pooling with k_pooling == {self.config.params.k_pooling}")
			main_layer = "pooled"  # if not in config.steps, use the main layer
			pp = PoissonPooling(self.config.params.k_pooling, self.config.params.n_genes, self.config.params.n_factors, compute_velocity=False, n_threads=self.config.execution.n_cpus, factorization=self.config.params.factorization, batch_keys=self.config.params.batch_keys)
			pp.fit_transform(ds)
		
		# Select features
		if self.config.params.features == "enrichment":
			logging.info(f"Feature selection by enrichment on preliminary clusters")
			with warnings.catch_warnings():
				warnings.simplefilter("ignore", category=NumbaPerformanceWarning)  # Suppress warnings about numba not being able to parallelize code
				warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)  # Suppress warnings about future deprecations
				warnings.simplefilter("ignore", category=SparseEfficiencyWarning)  # Suppress warnings about setting the diagonal to 1
				logging.info(f"  Gene selection for PCA")
				genes = FeatureSelectionByVariance(self.config.params.n_genes, mask=Species.mask(ds, self.config.params.mask)).fit(ds)
				logging.info(f"  Factorization by PCA")
				normalizer = Normalizer(False)
				normalizer.fit(ds)
				logging.info("  PCA projection to %d components", self.config.params.n_factors)
				pca = PCA(genes, max_n_components=self.config.params.n_factors, layer=main_layer, test_significance=False, batch_keys=self.config.params.batch_keys)
				transformed = pca.fit_transform(ds, normalizer)
				logging.info(f"  Computing KNN (k={self.config.params.k}) in PCA space")
				nn = NNDescent(data=transformed, metric="euclidean")
				indices, distances = nn.query(transformed, k=self.config.params.k)
				indices = indices[:, 1:]
				distances = distances[:, 1:]
				knn = sparse.csr_matrix(
					(np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])), (transformed.shape[0], transformed.shape[0])
				)

			g = nx.from_scipy_sparse_matrix(knn)
			partitions = community.best_partition(g, resolution=1, randomize=False)
			ds.ca.Clusters = np.array([partitions[key] for key in range(knn.shape[0])])
			n_labels = ds.ca.Clusters.max() + 1
			genes = FeatureSelectionByEnrichment(int(self.config.params.n_genes // n_labels), Species.mask(ds, self.config.params.mask), findq=False).select(ds)
		elif self.config.params.features == "variance":
			logging.info(f"Feature selection by variance")
			genes = FeatureSelectionByVariance(self.config.params.n_genes, main_layer, Species.mask(ds, self.config.params.mask)).select(ds)
		logging.info(f"Selected {genes.sum()} genes")

		if self.config.params.factorization in ['PCA', 'both']:
			logging.info(f"Factorization by PCA")
			normalizer = Normalizer(False)
			normalizer.fit(ds)
			n_components = min(self.config.params.n_factors, ds.shape[1])
			logging.info("  PCA projection to %d components", n_components)
			pca = PCA(genes, max_n_components=n_components, layer=main_layer, test_significance=False, batch_keys=self.config.params.batch_keys)
			ds.ca.PCA = pca.fit_transform(ds, normalizer)

		if self.config.params.factorization in ['HPF', 'both']:
			logging.info(f"Factorization by HPF")
			# Load the data for the selected genes
			data = ds[main_layer].sparse(rows=genes).T
			logging.debug(f"  Data shape is {data.shape}")

			# HPF factorization
			n_components = min(self.config.params.n_factors, round(ds.shape[1]/3))
			hpf = HPF(k=n_components, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False, n_threads=self.config.execution.n_cpus)
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

		if "nn" in self.config.steps or "clustering" in self.config.steps:
			if self.config.params.nn_space in ["PCA", "auto"] and "PCA" in ds.ca:
				transformed = ds.ca.PCA
				metric = "euclidean"
			elif self.config.params.nn_space in ["HPF", "auto"] and "HPF" in ds.ca:
				transformed = ds.ca.HPF
				metric = "js"
			logging.info(f"Computing balanced KNN (k = {self.config.params.k}) in {self.config.params.nn_space} space using the '{metric}' metric")
			bnn = BalancedKNN(k=self.config.params.k, metric=metric, maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
			bnn.fit(transformed)
			knn = bnn.kneighbors_graph(mode='distance')
			knn.eliminate_zeros()
			mknn = knn.minimum(knn.transpose())
			# Convert distances to similarities
			max_d = knn.data.max()
			knn.data = (max_d - knn.data) / max_d
			mknn.data = (max_d - mknn.data) / max_d
			ds.col_graphs.KNN = knn
			ds.col_graphs.MKNN = mknn
			mknn = mknn.tocoo()
			mknn.setdiag(0)
			# Compute the effective resolution
			d = 1 - knn.data
			radius = np.percentile(d, 90)
			logging.info(f"  90th percentile radius: {radius:.02}")
			ds.attrs.radius = radius
			inside = mknn.data > 1 - radius
			rnn = sparse.coo_matrix((mknn.data[inside], (mknn.row[inside], mknn.col[inside])), shape=mknn.shape)
			ds.col_graphs.RNN = rnn

		if "embeddings" in self.config.steps or "clustering" in self.config.steps:
			logging.info(f"Computing 2D and 3D embeddings from latent space")
			metric_f = (jensen_shannon_distance if metric == "js" else metric)  # Replace js with the actual function, since OpenTSNE doesn't understand js
			if transformed.shape[0] <= 200:
				ds.ca.TSNE = TSNE(perplexity=5).fit_transform(transformed)
				ds.ca.UMAP = UMAP(n_components=2, metric=metric_f).fit_transform(transformed)
				ds.ca.UMAP3D = UMAP(n_components=3, metric=metric_f).fit_transform(transformed)
			else:
				logging.info(f"  Art of tSNE with {metric} distance metric")
				ds.ca.TSNE = np.array(art_of_tsne(transformed, metric=metric_f))  # art_of_tsne returns a TSNEEmbedding, which can be cast to an ndarray (its actually just a subclass)
				logging.info(f"  UMAP with {metric} distance metric")
				ds.ca.UMAP = UMAP(n_components=2, metric=metric_f, n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(transformed)
				ds.ca.UMAP3D = UMAP(n_components=3, metric=metric_f, n_neighbors=self.config.params.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(transformed)

		if "clustering" in self.config.steps:
			if self.config.params.clusterer == "louvain":
				logging.info("Clustering by polished Louvain")
				pl = PolishedLouvain(outliers=False, graph=self.config.params.graph, embedding="TSNE")
				labels = pl.fit_predict(ds)
				ds.ca.Clusters = labels + min(labels)
				ds.ca.Outliers = (labels == -1).astype('int')
			elif self.config.params.clusterer == "sknetwork":
				logging.info("Clustering by unpolished scikit-network Louvain")
				G = nx.from_scipy_sparse_matrix(ds.col_graphs.KNN)
				adj = nx.linalg.graphmatrix.adjacency_matrix(G)
				labels = Louvain().fit_transform(adj)
				ds.ca.Clusters = labels + min(labels)
				ds.ca.Outliers = (labels == -1).astype('int')
			else:
				logging.info("Clustering by polished Surprise")
				ps = PolishedSurprise(graph=self.config.params.graph, embedding="TSNE")
				labels = ps.fit_predict(ds)
				ds.ca.Clusters = labels + min(labels)
				ds.ca.Outliers = (labels == -1).astype('int')
			logging.info(f"Found {ds.ca.Clusters.max() + 1} clusters")

		if species.name in ["Homo sapiens", "Mus musculus"]:
			logging.info(f"Inferring cell cycle")
			CellCycleAnnotator(species).annotate(ds)
