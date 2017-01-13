import numpy as np
import logging
import community
import networkx as nx
from scipy import sparse


class LouvainJaccard:
	def __init__(self, resolution: float = 1.0) -> None:
		self.graph = None  # type: nx.Graph
		self.resolution = resolution

	def fit_predict(self, knn: sparse.coo_matrix) -> np.ndarray:
		"""
		Given a sparse adjacency matrix, perform Louvain-Jaccard clustering

		Args:
			knn:	The sparse adjacency matrix

		Returns:
			labels:	The cluster labels

		Remarks:
			After clustering, the Louvain-Jaccard weighted undirected graph is available as 
			the property 'graph' of type nx.Graph
		"""
		edges = np.stack((knn.row, knn.col), axis=1)
		# Calculate Jaccard similarities
		js = []  # type: List[float]
		knncsr = knn.tocsr()
		for i, j in edges:
			r = knncsr.getrow(i)
			c = knncsr.getrow(j)
			shared = r.minimum(c).nnz
			total = r.maximum(c).nnz
			js.append(shared / total)
		weights = np.array(js) + 0.00001  # OpenOrd doesn't like 0 weights

		self.graph = nx.Graph()
		for i, edge in enumerate(edges):
			self.graph.add_edge(edge[0], edge[1], {'weight': weights[i]})

		partitions = community.best_partition(self.graph, resolution=self.resolution)
		return np.fromiter(partitions.values(), dtype='int')
