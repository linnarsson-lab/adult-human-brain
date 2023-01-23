import loompy
import numpy as np
import sys
from cytograph.pipeline import load_config
import networkx as nx
import community
from sknetwork.hierarchy import Paris, cut_straight

subset = sys.argv[1]
n_cut = int(sys.argv[2])
config = load_config()

def calc_cpu(n_cells):
    n = np.array([1e2, 1e3, 1e4, 1e5, 5e5, 1e6, 2e6])
    cpus = [1, 3, 7, 14, 28, 28, 56]
    idx = (np.abs(n - n_cells)).argmin()
    return cpus[idx]

with loompy.connect(f'data/{subset}.loom') as ds:

	print("Loading KNN graph")
	G = nx.from_scipy_sparse_matrix(ds.col_graphs.KNN)
	print("Partitioning graph by Cytograph clusters")
	partition = dict(zip(np.arange(ds.shape[1]), ds.ca.Clusters))
	print("Generating induced adjacency  matrix")
	induced = community.induced_graph(partition, G)
	adj = nx.linalg.graphmatrix.adjacency_matrix(induced)
	print("Paris clustering")
	Z = Paris().fit_transform(adj)
	ds.attrs.paris_linkage = Z

	print(f"Cutting dendrogram into {n_cut} groups")
	clusters = cut_straight(Z, n_cut)[ds.ca.Clusters]
	ds.ca[f'Paris{n_cut}'] = clusters
	ds.ca.Split = clusters

	# Calculate split sizes
	sizes = np.bincount(clusters)
	print("Creating punchcard")
	with open(f'punchcards/{subset}.yaml', 'w') as f:
		for i in np.unique(clusters):
			# Calc cpu and memory
			n_cpus = calc_cpu(sizes[i])
			memory = 750 if n_cpus == 56 else config.execution.memory
			# Write to punchcard
			name = chr(i + 65) if i < 26 else chr(i + 39) * 2
			f.write(f'{name}:\n')
			f.write('  include: []\n')
			f.write(f'  onlyif: Split == {i}\n')
			f.write('  execution:\n')
			f.write(f'    n_cpus: {n_cpus}\n')
			f.write(f'    memory: {memory}\n')