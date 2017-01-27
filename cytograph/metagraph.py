
import loompy
import numpy as np
from scipy import sparse
import networkx as nx


class MetaGraph:
	def __init__(self) -> None:
		pass

	def make(self, knn: sparse.coo_matrix, labels: np.ndarray) -> nx.Graph:
		knn_csr = knn.tocsr()
		g = nx.Graph()
		for lbl in range(0, max(labels) + 1):
			size = np.sum(labels == lbl)
			if size == 0:
				continue
			node = g.add_node(lbl, size=size)
		for a in range(0, max(labels) + 1):
			for b in range(0, max(labels) + 1):
				# how many knn nodes link between these nodes?
				g.add_edge(a, b, weight=knn_csr[labels == a, :][:, labels == b].sum())
		return g
