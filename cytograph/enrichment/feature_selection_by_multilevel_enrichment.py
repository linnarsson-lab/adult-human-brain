import logging
from typing import List

import numpy as np
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
from scipy import sparse

import loompy


class FeatureSelectionByMultilevelEnrichment:
	"""
	Find markers at each of several levels relative to cluster labels
	"""
	def __init__(self, n_clusters_per_level: List[int] = None, n_markers_per_cluster: int = 10, labels_attr: str = "Clusters", mask: np.ndarray = None) -> None:
		"""
		Args:
			n_clusters_per_level		Desired number of clusters at each levels above the leaves, which determines where the tree is cut
			n_markers_per_cluster		Number of markers to include per cluster
		"""
		self.n_clusters_per_level = n_clusters_per_level
		self.n_markers_per_cluster = n_markers_per_cluster
		self.labels_attr = labels_attr
		self.mask = mask
		self.valid_genes: np.ndarray = None
		self.enrichment: np.ndarray = None

	def fit(self, ds: loompy.LoomConnection, labels: np.ndarray = None) -> np.ndarray:
		"""
		Finds n_markers genes per cluster using enrichment score, at each of several levels

		Args:
			ds (LoomConnection):	Dataset
			labels					Optional labels to use instead of the cluster labels

		Returns:
			ndarray of selected marker genes (array of ints), shape (n_markers)
			ndarray of enrichment scores for the leaf level only, shape (n_genes, n_labels)
		"""
		n_genes, n_cells = ds.shape

		# Find the cluster labels
		if labels is None:
			labels = ds.ca[self.labels_attr]
		n_labels = len(np.unique(labels))
		logging.info(f"Multilevel marker selection with {n_labels} clusters at the leaf level")

		# Find a good set of levels
		if self.n_clusters_per_level is None:
			proposal = np.array([25, 10, 5, 2])
			proposal = proposal[proposal < n_labels // 2]
			self.n_clusters_per_level = list(proposal)
		n_levels = len(self.n_clusters_per_level)
		if n_levels > 0:
			logging.info(f"Analyzing {n_levels} higher level{'s' if n_levels > 1 else ''} with {self.n_clusters_per_level} clusters")
			multilevel_markers = np.zeros((ds.shape[0], n_levels))

			# Find markers at the leaf level
			(all_markers, all_enrichment, means) = self._fit(ds, labels)
			logging.info(f"Found {all_markers.sum()} marker genes at level 0 (leaves)")

			# Agglomerative clustering
			data = np.log(means + 1)[all_markers, :].T
			D = pdist(data, 'correlation')
			Z = hc.linkage(D, 'ward', optimal_ordering=True)
			old_labels_per_cluster = hc.leaves_list(Z)
			old_labels_per_cell = labels.copy()

			# Select markers at each level
			i = 0
			while i < n_levels:
				new_labels_per_cluster = hc.cut_tree(Z, n_clusters=self.n_clusters_per_level[i])
				temp = np.zeros_like(labels)
				for lbl in np.unique(old_labels_per_cluster):
					temp[old_labels_per_cell == lbl] = new_labels_per_cluster[old_labels_per_cluster == lbl][0]
				labels = temp
				(markers, enrichment, _) = self._fit(ds, labels)
				logging.info(f"Found {markers.sum()} marker genes at level {i + 1}")
				logging.debug(ds.ra.Gene[markers])
				multilevel_markers[markers, i] = 1
				all_markers = (all_markers | markers)
				i += 1
			ds.ra.MultilevelMarkers = multilevel_markers
		else:
			logging.info("Not enough clusters for multilevel marker selection (using level 0 markers only)")
			# Find markers at the leaf level
			(all_markers, all_enrichment, means) = self._fit(ds, labels)
			logging.info(f"Found {all_markers.sum()} marker genes at level 0 (leaves)")

		self.enrichment = all_enrichment
		selected = np.zeros(ds.shape[0], dtype=bool)
		selected[np.where(all_markers)[0]] = True
		return selected

	def select(self, ds: loompy.LoomConnection) -> np.ndarray:
		selected = self.fit(ds)
		ds.ra.Selected = selected.astype("int")
		return selected

	def _fit(self, ds: loompy.LoomConnection, labels: np.ndarray) -> np.ndarray:
		logging.info("Computing enrichment statistic")
		n_labels = len(np.unique(labels))
		n_genes, n_cells = ds.shape

		# reorder matrix
		mtx = ds.sparse().tocsr()
		ordering = np.argsort(labels)
		arr_list = []
		chunksize = 100000000 // mtx.shape[1]
		start = 0
		while start < mtx.shape[0]:
			submatrix = mtx[start:start + chunksize, :]
			arr_list.append(submatrix[:, ordering])
			start = start + chunksize
		mtx = sparse.vstack(arr_list)

		# split matrix on label indices
		ix = [np.where(np.sort(labels) == x)[0][0] for x in range(n_labels)] + [n_cells]
		arr_list = []
		for i in range(len(ix) - 1):
			arr_list.append(mtx[:, ix[i]:ix[i + 1]])

		# Number of cells per cluster
		sizes = np.zeros(n_labels)
		# Number of nonzero values per cluster
		nnz = np.zeros((ds.shape[0], n_labels))
		# Mean value per cluster
		means = np.zeros((ds.shape[0], n_labels))

		for i, arr in enumerate(arr_list):
			nnz[:, i] = arr.getnnz(axis=1)
			means[:, i] = np.squeeze((arr.mean(axis=1).A))
			sizes[i] = arr.shape[1]

		# Non-zeros and means over all cells
		mtx = sparse.hstack(arr_list)
		nnz_overall = mtx.getnnz(axis=1)
		means_overall = np.squeeze((mtx.mean(axis=1).A))

		# Scale by number of cells
		n_cells = mtx.shape[1]
		f_nnz = nnz / sizes
		f_nnz_overall = nnz_overall / n_cells

		# Means and fraction non-zero values in other clusters (per cluster)
		means_other = ((means_overall * n_cells)[None].T - (means * sizes)) / (n_cells - sizes)
		f_nnz_other = ((f_nnz_overall * n_cells)[None].T - (f_nnz * sizes)) / (n_cells - sizes)

		# enrichment = (f_nnz + 0.1) / (f_nnz_overall[None].T + 0.1) * (means + 0.01) / (means_overall[None].T + 0.01)
		enrichment = (f_nnz + 0.1) / (f_nnz_other + 0.1) * (means + 0.01) / (means_other + 0.01)

		# Select best markers
		if self.valid_genes is None:
			logging.info("Identifying valid genes")
			if "Valid" in ds.ra:
				self.valid_genes = ds.ra.Valid == 1
			else:
				nnz = ds.map([np.count_nonzero], axis=0)[0]
				self.valid_genes = np.logical_and(nnz > 10, nnz < ds.shape[1] * 0.6)

		if self.mask is None:
			excluded = set(np.where(~self.valid_genes)[0])
		else:
			excluded = set(np.where(((~self.valid_genes) | self.mask))[0])

		# Handle datasets with thousands of clusters	
		if n_genes - len(excluded) < self.n_markers_per_cluster * n_labels:
			included = self.valid_genes & ~self.mask
			return (included, enrichment, means)

		included = np.zeros(n_genes, dtype=bool)
		for ix in range(n_labels):
			enriched = np.argsort(enrichment[:, ix])[::-1]
			n = 0
			count = 0
			while count < self.n_markers_per_cluster:
				if enriched[n] in excluded:
					n += 1
					continue
				included[enriched[n]] = True
				excluded.add(enriched[n])
				n += 1
				count += 1
		return (included, enrichment, means)
