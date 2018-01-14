import numpy as np
import scipy.sparse as sparse
from sklearn.neighbors import NearestNeighbors
import logging
import loompy
import cytograph as cg
import logging


class PoissonImputation:
	def __init__(self, k: int = 5, n_genes: int = 1000, n_components: int = 50) -> None:
		self.k = k
		self.n_genes = n_genes
		self.n_components = n_components

	def impute_inplace(self, ds: loompy.LoomConnection) -> np.ndarray:
		logging.info("Normalization")
		normalizer = cg.SqrtNormalizer()
		normalizer.fit(ds)

		# Select genes
		genes = cg.FeatureSelection(n_genes=self.n_genes).fit(ds)

		logging.info("PCA projection to %d components", self.n_components)
		pca = cg.PCAProjection(genes, max_n_components=self.n_components)
		transformed = pca.fit_transform(ds, normalizer)

		# Compute KNN matrix
		np.random.seed(0)
		logging.info("Computing nearest neighbors")
		nn = NearestNeighbors(self.k, algorithm="auto", metric='correlation', n_jobs=4)
		nn.fit(transformed)
		knn = nn.kneighbors_graph(transformed, mode='connectivity')  # Returns a CSR sparse graph, including self-edges

		# Compute size-corrected Poisson MLE rates
		size_factors = ds.map([np.sum], axis=1)[0]

		logging.info("Imputing values in place")
		ix = 0
		window = 400
		while ix < ds.shape[0]:
			# Load the data for a subset of genes
			data = ds[ix:min(ds.shape[0], ix + window), :]
			data_std = data / size_factors
			# Sum of MLE rates for neighbors
			imputed = knn.dot(data_std.T).T
			ds[ix:min(ds.shape[0], ix + window), :] = imputed.astype('float32')
			ix += window
