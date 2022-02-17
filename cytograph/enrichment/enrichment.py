import numpy as np
import loompy
import numpy_groupies as npg


class Enrichment:
	"""
	Compute gene enrichment across clusters on already aggregated data
	"""
	def __init__(self) -> None:
		self.nonzeros = None

	def fit(self, dsout: loompy.LoomConnection, ds: loompy.LoomConnection) -> np.ndarray:
		"""
		Compute gene enrichment across clusters on already aggregated data
		
		Args:
			ds		A loom file

		Returns:
			enrichment		A (n_genes, n_clusters) matrix of gene enrichment scores
		"""

		n_clusters = dsout.shape[1]
		cluster_size = dsout.ca.NCells
		means = dsout[:, :]  # / cluster_size * np.median(cluster_size)
		self.nonzeros = self.cluster_nonzeros(ds)
		f_nnz = self.nonzeros / cluster_size

		# calculate enrichment scores
		enrichment = np.zeros_like(means)
		for j in range(n_clusters):
			# calculate cluster weights
			ix = np.arange(n_clusters) != j
			weights = cluster_size[ix] / cluster_size[ix].sum()
			# calculate means_other as weighted average
			means_other = np.average(means[:, ix], weights=weights, axis=1)
			# calculate f_nnz as weighted average
			f_nnz_other = np.average(f_nnz[:, ix], weights=weights, axis=1)
			# calculate enrichment
			enrichment[:, j] = (f_nnz[:, j] + 0.1) / (f_nnz_other + 0.1) * (means[:, j] + 0.01) / (means_other + 0.01)

		return enrichment

	def cluster_nonzeros(self, ds: loompy.LoomConnection) -> np.ndarray:
		"""
		Compute fraction of cells in a cluster that are nonzero for each gene.
		
		Args:
			ds		A loom file

		Returns:
			enrichment		A (n_genes, n_clusters) matrix of nonzero fractions
		"""

		batch_size = 1000
		n_clusters = ds.ca.Clusters.max() + 1
		nonzeros = np.empty((ds.shape[0], n_clusters))

		for ix in range(0, ds.shape[0], batch_size):
			vals = ds[ix:ix + batch_size, :]
			nnz = npg.aggregate(ds.ca.Clusters, vals > 0, func='sum', axis=1)
			nonzeros[ix:ix + batch_size, :] = nnz

		return nonzeros
