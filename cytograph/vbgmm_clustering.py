from typing import *
import logging
from sklearn.mixture import BayesianGaussianMixture as BGM
import numpy as np
import loompy
import cytograph as cg


class VbgmmClustering:
	def __init__(self, max_clusters: int = 100, gamma_multiplier: int = 1) -> None:
		self.max_clusters = max_clusters
		self.gamma = gamma_multiplier / max_clusters
	
	def fit_predict(self, ds: loompy.LoomConnection) -> np.ndarray:
		n_cells = ds.shape[0]
		logging.info(f"VBGMM on all {n_cells} cells")
		
		if "_Valid" in ds.ra:
			genes = (ds.ra._Valid == 1)
		else:
			logging.info(f"Computing valid genes")
			nnz = ds.map([np.count_nonzero], axis=0)[0]
			genes = np.logical_and(nnz > 10, nnz < ds.shape[1] * 0.6).astype('int')
			ds.ra._Valid = genes
		n_genes = genes.sum()
		logging.info(f"Using {n_genes} of {ds.shape[0]} genes")

		logging.info(f"Normalizing")
		normalizer = cg.Normalizer(True)
		normalizer.fit(ds)

		n_components = min(1000, n_cells)
		logging.info("PCA projection to %d components", n_components)
		pca = cg.PCAProjection(genes, max_n_components=n_components)
		transformed = pca.fit_transform(ds, normalizer)

		logging.info(f"Finding up to {self.max_clusters} clusters by VBGMM, using gamma={self.gamma}")
		labels = BGM(n_components=self.max_clusters, weight_concentration_prior=self.gamma).fit(transformed).predict(transformed)
		logging.info(f"Found {np.max(labels) + 1} clusters")
		
		return labels

# TODO: use the cell clusters as features for genes, then cluster genes to describe the modular composition of cell types
# TODO: use random forest classifier to assess the quality of clusters, and use it to find a good resolution
# TODO: use predict_proba to define outliers
# TODO: use predict_proba to define graded cluster memberships
# TODO: use predict_proba to create a metagraph of clusters, with edges representing secondary label probabilities

