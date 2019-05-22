import loompy
import scipy.sparse as sparse
import numpy as np


class GraphSkeletonizer:
	"""
	Implements a simple form of graph skeletonization. Based on the cluster labels of individual cells,
	constructs a graph with clusters as vertices, and connecting each pair of clusters where the
	number of edges between the two clusters (in the underlying RNN graph) exceeds min_pct percent
	of the total number of edges in the two clusters.

	Uses the RNN graph if present, otherwise the KNN graph.
	"""
	def __init__(self, min_pct: float = 2) -> None:
		self.min_pct = min_pct
	
	def fit(self, ds: loompy.LoomConnection) -> sparse.coo_matrix:
		row = []
		col = []
		weight = []
		nn = ds.col_graphs.RNN if "RNN" in ds.col_graphs else ds.col_graphs.KNN
		for c1 in np.unique(ds.ca.Clusters):
			n_c1 = (ds.ca.Clusters == c1).sum()
			rc1 = ds.ca.Clusters[nn.row] == c1
			cc1 = ds.ca.Clusters[nn.col] == c1
			c1_internal = (rc1 & cc1).sum()
			for c2 in np.unique(ds.ca.Clusters):
				if c2 <= c1:
					continue
				n_c2 = (ds.ca.Clusters == c2).sum()
				rc2 = ds.ca.Clusters[nn.row] == c2
				cc2 = ds.ca.Clusters[nn.col] == c2
				c2_internal = (rc2 & cc2).sum()
				c1c2_between = ((rc1 & cc2) | (rc2 & cc1)).sum()
				f = n_c1 / (n_c1 + n_c2)
				expected_fraction = (2 * f - 2 * f**2)
				observed_fraction = c1c2_between / (c1_internal + c2_internal + c1c2_between)
				# if observed_fraction > expected_fraction * (self.min_pct / 100):
				if observed_fraction > (self.min_pct / 100):
					row.append(c1)
					col.append(c2)
					weight.append(observed_fraction / expected_fraction)
		n_clusters = ds.ca.Clusters.max() + 1
		return sparse.coo_matrix((weight, (row, col)), shape=(n_clusters, n_clusters))
	
	def abstract(self, ds: loompy.LoomConnection, dsagg: loompy.LoomConnection) -> sparse.coo_matrix:
		"""
		Compute the graph abstraction and save a column graph named "GA" in the dsagg loom file. Also
		compute median values of TSNE and UMAP in the ds file and save in dsagg. These can be used
		as the location of the vertices of the abstracted graph.
		"""
		dsagg.col_graphs.GA = self.fit(ds)
		dsagg.ca.TSNE = np.array([np.median(ds.ca.TSNE[ds.ca.Clusters == c], axis=0) for c in range(ds.ca.Clusters.max() + 1)])
		dsagg.ca.UMAP = np.array([np.median(ds.ca.UMAP[ds.ca.Clusters == c], axis=0) for c in range(ds.ca.Clusters.max() + 1)])
