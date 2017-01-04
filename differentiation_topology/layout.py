from subprocess import check_output, CalledProcessError
from typing import *
import tempfile
import os
import shutil
import logging
import numpy as np
import networkx as nx
from scipy import sparse


class SFDP:
	def __init__(self, sfdp_path: str = "") -> None:
		self.sfdp_path = sfdp_path
		self.graph = None  # type: nx.Graph

	def layout_knn(self, knn: sparse.coo_matrix) -> np.ndarray:
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
		self.graph.add_nodes_from(range(knn.shape[0]))
		for i, edge in enumerate(edges):
			self.graph.add_edge(edge[0], edge[1], {'weight': weights[i]})

		return self.layout(self.graph)

	def layout(self, graph: nx.Graph) -> np.ndarray:
		"""
		Use SFDP to compute a graph layout

		Remarks:
			Requires Graphviz to be installed and 'sfdp' available in the $PATH
		"""
		tempdir = tempfile.mkdtemp()
		# Save the graph in .int format
		infile = os.path.join(tempdir, "graph.dot")
		outfile = os.path.join(tempdir, "graph_sfdp.dot")
		nx.nx_agraph.write_dot(graph, infile)
		try:
			_ = check_output([os.path.join(self.sfdp_path, "sfdp"), infile, "-o" + outfile])
		except CalledProcessError as e:
			shutil.rmtree(tempdir)
			raise e

		# Read back the coordinates
		def parse_pos(s: str) -> List[float]:
			items = s.split(",")
			if len(items) < 2:
				return [0, 0]
			return [float(items[0]), float(items[1])]

		g = nx.nx_agraph.read_dot(outfile)
		coords = np.array([[float(n)] + parse_pos(d['pos']) for n, d in g.nodes_iter(data=True)])
		# Sort by first column, then take last two columns
		coords = coords[coords[:, 0].argsort()][:, -2:]
		shutil.rmtree(tempdir)
		return coords


class OpenOrd:
	"""
	Implements an interface to the OpenOrd command-line tool.
	See http://www.sandia.gov/~smartin/software.html
	"""
	def __init__(self, openord_path: str = "", edge_cutting: float = 0.8) -> None:
		"""
		Create an OpenOrd object with the given parameters
		"""
		self.openord_path = openord_path
		self.edge_cutting = edge_cutting

	def layout_knn(self, knn: sparse.coo_matrix) -> np.ndarray:
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
		self.graph.add_nodes_from(range(knn.shape[0]))
		for i, edge in enumerate(edges):
			self.graph.add_edge(edge[0], edge[1], {'weight': weights[i]})

		return self.layout(self.graph)

	def layout(self, graph: nx.Graph) -> np.ndarray:
		"""
		Use OpenOrd to compute a graph layout

		Remarks:
			Requires OpenOrd to be installed and available in the $PATH
		"""
		tempdir = tempfile.mkdtemp()
		# Save the graph in .int format
		rootfile = os.path.join(tempdir, "graph")
		with open(rootfile + ".int", 'w') as f:
			for (x, y, w) in graph.edges_iter(data='weight'):
				f.write("%d\t%d\t%f\n" % (x, y, w))
		try:
			_ = check_output([os.path.join(self.openord_path, "layout"), "-c", str(self.edge_cutting), rootfile])
		except CalledProcessError as e:
			shutil.rmtree(tempdir)
			raise e
		# Read back the coordinates
		with open(rootfile + ".icoord", 'r') as f:
			coords = np.zeros((graph.number_of_nodes(), 2))
			for line in f:
				items = line.split("\t")
				node = int(items[0])
				coords[node] = (float(items[1]), float(items[2]))
		shutil.rmtree(tempdir)
		return coords
