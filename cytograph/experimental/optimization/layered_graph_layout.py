import cvxpy as cp
import numpy as np
from typing import List


class LayeredGraphLayout:
	"""
	Layout a layered graph with edge weights so as to minimize weighted edge crossings
	
	Remarks:
		Based on the algorithm by Zarate et al. https://ieeexplore.ieee.org/document/8365985
	"""
	def __init__(self) -> None:
		pass
	
	def fit(self, n_nodes: List[int], edges: List[np.ndarray]) -> List[np.ndarray]:
		"""
		Args:
			n_nodes		List of node counts per layer
			edges		List of edge matrices, each an array shaped (n_edges, 3) where each row is an (i, j, w) tuple
						giving the starting node i, ending node j, and weight w of the edge.
		"""
		n_edges = [x.shape[0] for x in edges]
		n_layers = len(n_nodes)

		# Edge weight overlap areas
		w_areas = [np.outer(edges[i][:, 2], edges[i][:, 2].T) for i in range(len(n_edges))]
		for w in w_areas:
			np.fill_diagonal(w, 0)  # Crossing of edge with itself is given zero weight

		# Crossing indicators
		c_vars = []
		for e in n_edges:
			c_vars.append(cp.Variable(shape=(e, e), boolean=True))

		# Position indicators
		x_vars = []
		for n in n_nodes:
			x_vars.append(cp.Variable(shape=(n, n), boolean=True))

		# Objective function
		for layer in range(n_layers - 1):
			obj = cp.sum(w_areas[layer] * c_vars[layer])

		# Constraints
		constraints = []
		for layer in range(n_layers):
			# We subtract the diagonal, because otherwise there will be no solution (1 + 1 = 2 or 0 + 0 = 0)
			constraints.append(x_vars[layer] + x_vars[layer].T - cp.diag(cp.diag(x_vars[layer])) == 1)

		for x in x_vars:
			# For each triplet of nodes (skipping equal nodes)
			for i in range(x.shape[0]):
				for j in range(x.shape[0]):
					if i == j:
						continue
					for k in range(x.shape[0]):
						if i == k or j == k:
							continue
						constraints.append(x[k, i] >= x[k, j] + x[j, i] - 1)

		for layer in range(n_layers - 1):
			# For each pair of edges in the layer
			for i in range(c_vars[layer].shape[0]):
				for j in range(c_vars[layer].shape[0]):
					# Edge i crosses edge j iff node i0 is above node j0 and node i1 is below node j1, or the other way around
					u1 = edges[layer][i, 0]  # The starting node of edge i
					v1 = edges[layer][i, 1]  # The ending node of edge i
					u2 = edges[layer][j, 0]  # The starting node of edge j
					v2 = edges[layer][j, 1]  # The ending node of edge j
					constraints.append(c_vars[layer][i, j] + x_vars[layer][u2, u1] + x_vars[layer + 1][v1, v2] >= 1)
					constraints.append(c_vars[layer][i, j] + x_vars[layer][u1, u2] + x_vars[layer + 1][v2, v1] >= 1)

		# Additional constraints to help convergence
		for layer in range(n_layers - 1):
			# If the edges i and j cross, then edges j and i also cross
			constraints.append(c_vars[layer] == c_vars[layer].T)

		prob = cp.Problem(cp.Minimize(obj), constraints)
		prob.solve()
		orderings = [x.value.sum(axis=0) - 1 for x in x_vars]
		return [x.astype("int") for x in orderings]
