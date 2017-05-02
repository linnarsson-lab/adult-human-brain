import numpy as np
import logging
import community
import networkx as nx
from scipy import sparse
from typing import *


class LouvainJaccard:
	def __init__(self, resolution: float = 1.0, jaccard: bool = True) -> None:
		self.graph = None  # type: nx.Graph
		self.resolution = resolution
		self.jaccard = jaccard

	def fit_predict(self, knn: sparse.coo_matrix) -> np.ndarray:
		"""
		Given a sparse adjacency matrix, perform Louvain-Jaccard clustering

		Args:
			knn:	The sparse adjacency matrix

		Returns:
			labels:	The cluster labels

		Remarks:
			After clustering, the Louvain-Jaccard weighted undirected graph is available as
			the property 'graph' of type nx.Graph, and also in the form of a sparse adjacency
			matrix as the property 'lj_knn' of type scipy.sparse.coo_matrix
		"""
		if self.jaccard:
			edges = np.stack((knn.row, knn.col), axis=1)
			# Calculate Jaccard similarities
			js = []  # type: List[float]
			knncsr = knn.tocsr()
			for i, j in edges:
				r = knncsr.getrow(i)
				c = knncsr.getrow(j)
				shared = r.minimum(c).nnz
				total = r.maximum(c).nnz
				if total > 0:
					js.append(shared / total)
				else:
					js.append(0)
			weights = np.array(js) + 0.00001  # OpenOrd doesn't like 0 weights

			self.lj_knn = sparse.coo_matrix((weights, (knn.row, knn.col)))
			self.graph = nx.Graph()
			for i, edge in enumerate(edges):
				self.graph.add_edge(edge[0], edge[1], {'weight': weights[i]})
		else:
			self.graph = nx.from_scipy_sparse_matrix(knn)
		partitions = community.best_partition(self.graph, resolution=self.resolution)
		return np.array([partitions[key] for key in range(knn.shape[0])])
