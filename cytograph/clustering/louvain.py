import logging
import community
import leidenalg
import networkx as nx
import igraph as ig
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
import loompy


class Louvain:
	def __init__(self, resolution: float = 1.0, min_cells: int = 10, graph: str = "MKNN", embedding: str = "TSNE", method: str = "python-louvain") -> None:
		self.resolution = resolution
		self.min_cells = min_cells
		self.graph = graph
		self.embedding = embedding
		self.method = method  # "leiden" or "python-louvain"

	def fit_predict(self, ds: loompy.LoomConnection) -> np.ndarray:
		"""
		Given a sparse adjacency matrix, perform Louvain clustering, then polish the result

		Args:
			ds		The loom dataset
			graph	The graph to use

		Returns:
			labels:	The cluster labels (where -1 indicates outliers)

		"""
		if self.embedding in ds.ca:
			xy = ds.ca[self.embedding]
		else:
			raise ValueError(f"Embedding '{self.embedding}' not found in file")

		knn = ds.col_graphs[self.graph]

		logging.info("Louvain community detection")
		if self.method == "leiden":
			_, components = connected_components(knn)
			next_label = 0
			all_labels = np.zeros(ds.shape[1], dtype="int")
			for cc in np.unique(components):
				# Create an RNN graph restricted to the component
				cells = np.where(components == cc)[0]
				n_cells = cells.shape[0]
				if n_cells > self.min_cells:
					cc_rnn = ds.col_graphs[cells].RNN.tocsr()
					sources, targets = cc_rnn.nonzero()
					weights = cc_rnn[sources, targets]
					g = ig.Graph(list(zip(sources, targets)), directed=False, edge_attrs={'weight': weights})
					labels = np.array(leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, n_iterations=-1).membership) + next_label
				else:
					labels = np.zeros(n_cells) + next_label
				next_label = labels.max() + 1
				all_labels[cells] = labels
			labels = all_labels
		else:
			g = nx.from_scipy_sparse_matrix(knn)
			partitions = community.best_partition(g, resolution=self.resolution, randomize=False)
			labels = np.array([partitions[key] for key in range(knn.shape[0])])

		# Mark tiny clusters as outliers
		logging.info("Marking tiny clusters as outliers")
		ix, counts = np.unique(labels, return_counts=True)
		labels[np.isin(labels, ix[counts < self.min_cells])] = -1

		# Renumber the clusters (since some clusters might have been lost in poor neighborhoods)
		retain = list(set(labels))
		if -1 not in retain:
			retain.append(-1)
		retain = sorted(retain)
		d = dict(zip(retain, np.arange(-1, len(set(retain)))))
		labels = np.array([d[x] if x in d else -1 for x in labels])

		if np.all(labels < 0):
			logging.warn("All cells were determined to be outliers!")
			return np.zeros_like(labels)

		if np.any(labels == -1):
			# Assign each outlier to the same cluster as the nearest non-outlier
			nn = NearestNeighbors(n_neighbors=50, algorithm="ball_tree")
			nn.fit(xy[labels >= 0])
			nearest = nn.kneighbors(xy[labels == -1], n_neighbors=1, return_distance=False)
			labels[labels == -1] = labels[labels >= 0][nearest.flat[:]]

		return labels
