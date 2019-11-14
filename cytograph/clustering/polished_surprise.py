import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

import loompy


class PolishedSurprise:
	def __init__(self, min_cells: int = 20, graph: str = "RNN", embedding: str = "TSNE") -> None:
		self.min_cells = min_cells
		self.graph = graph
		self.embedding = embedding

	def fit_predict(self, ds: loompy.LoomConnection) -> np.ndarray:
		rnn = ds.col_graphs[self.graph]
		n_components, components = connected_components(rnn)
		next_label = 0
		all_labels = np.zeros(ds.shape[1], dtype="int")
		for cc in np.unique(components):
			# Create an RNN graph restricted to the component
			cells = np.where(components == cc)[0]
			n_cells = cells.shape[0]
			if n_cells > self.min_cells:
				cc_rnn = ds.col_graphs[cells][self.graph].tocsr()
				sources, targets = cc_rnn.nonzero()
				weights = cc_rnn[sources, targets]
				g = ig.Graph(list(zip(sources, targets)), directed=False, edge_attrs={'weight': weights})
				labels = np.array(leidenalg.find_partition(g, leidenalg.SurpriseVertexPartition).membership) + next_label
			else:
				labels = np.zeros(n_cells) + next_label
			next_label = labels.max() + 1
			all_labels[cells] = labels

		# Reassign each cluster to the mode of their neighbors
		xy = ds.ca[self.embedding]
		nn = NearestNeighbors(n_neighbors=10, algorithm="ball_tree", n_jobs=4)
		nn.fit(xy)
		knn = nn.kneighbors_graph(mode='connectivity').tocoo()
		temp = []
		for ix in range(all_labels.shape[0]):
			neighbors = knn.col[np.where(knn.row == ix)[0]]
			temp.append(mode(all_labels[neighbors])[0][0])
		all_labels = np.array(temp)
		
		tiny = np.isin(all_labels, np.where(np.bincount(all_labels) < self.min_cells)[0])
		if tiny.shape[0] > 0:
			all_labels[tiny] = -1
			nn = NearestNeighbors(n_neighbors=50, algorithm="ball_tree")
			nn.fit(xy[all_labels >= 0])
			nearest = nn.kneighbors(xy[all_labels == -1], n_neighbors=1, return_distance=False)
			all_labels[all_labels == -1] = all_labels[all_labels >= 0][nearest.flat[:]]

		# Renumber the clusters (since some clusters might have been lost in poor neighborhoods)
		all_labels = LabelEncoder().fit_transform(all_labels)
		return all_labels
