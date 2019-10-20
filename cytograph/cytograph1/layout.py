from subprocess import check_output, CalledProcessError, Popen
from typing import *
import tempfile
import os
from struct import calcsize, pack, unpack
import shutil
import logging
import numpy as np
import networkx as nx
from scipy import sparse


class SFDP:
	def __init__(self) -> None:
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
			_ = check_output(["sfdp", infile, "-o" + outfile])
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


def _read_unpack(fmt: str, fh: Any) -> Any:
	return unpack(fmt, fh.read(calcsize(fmt)))


class TSNE:
	def __init__(self, theta: float = 0.5, perplexity: float = 50, n_dims: int = 2, max_iter: int = 1000) -> None:
		self.theta = theta
		self.perplexity = perplexity
		self.n_dims = n_dims
		self.max_iter = max_iter
	
	def layout(self, transformed: np.ndarray, initial_pos: np.ndarray = None, knn: sparse.csr_matrix = None) -> np.ndarray:
		"""
		Compute Barnes-Hut approximate t-SNE layout

		Args:
			transformed:	The (typically) PCA-transformed input data, shape: (n_samples, n_components)
			n_dims:			2 or 3
			initial_pos:	Initial layout, or None to use the first components of 'transformed'
			knn: 			Precomputed knn graph in sparse matrix format, or None to use Gaussian perplexity

		Remarks:
			Requires 'bh_tsne' to be available on the $PATH
		"""
		n_cells = transformed.shape[0]
		n_components = transformed.shape[1]
		nnz = 0
		if initial_pos is None:
			initial_pos = transformed[:, :self.n_dims]
		if knn is not None:
			# knn = knn.tocsr().maximum(knn.transpose())
			# knn = knn.multiply(1 / knn.sum(axis=1)).tocsr()
			knn.sort_indices()
			nnz = knn.nnz
		with tempfile.TemporaryDirectory() as td:
			with open(os.path.join(td, 'data.dat'), 'wb') as data_file:
				# Write the bh_tsne header
				data_file.write(pack('=iiiddii', n_cells, n_components, nnz, self.theta, self.perplexity, self.n_dims, self.max_iter))
				# Write the initial positions
				for ix in range(n_cells):
					pos = initial_pos[ix, :]
					data_file.write(pack('={}d'.format(pos.shape[0]), *pos))
				if nnz != 0:
					data_file.write(pack('={}i'.format(knn.indptr.shape[0]), *knn.indptr))
					data_file.write(pack('={}i'.format(knn.indices.shape[0]), *knn.indices))
					data_file.write(pack('={}d'.format(knn.data.shape[0]), *knn.data))
				# Then write the data
				for ix in range(n_cells):
					sample = transformed[ix, :]
					data_file.write(pack('={}d'.format(sample.shape[0]), *sample))

			# Call bh_tsne and let it do its thing
			with open(os.devnull, 'w') as dev_null:
				bh_tsne_p = Popen(("bh_tsne", ), cwd=td)
				bh_tsne_p.wait()
				if bh_tsne_p.returncode != 0:
					logging.error("TSNE layout failed to execute external binary 'bh_tsne' (check $PATH)")
					raise RuntimeError()

			# Read and pass on the results
			with open(os.path.join(td, 'result.dat'), 'rb') as output_file:
				# The first two integers are just the number of samples and the
				#   dimensionality
				_, n_dims = _read_unpack('ii', output_file)
				# Collect the results, but they may be out of order
				results = [_read_unpack('{}d'.format(n_dims), output_file) for _ in range(n_cells)]
				# Now collect the landmark data so that we can return the data in
				#   the order it arrived
				results = [(_read_unpack('i', output_file), e) for e in results]
				# Put the results in order and yield it
				results.sort()
				xy = [result for _, result in results]
				return np.array(xy)


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
