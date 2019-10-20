from typing import Union, Tuple

import numpy as np
import scipy.sparse as sparse
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import NearestNeighbors

import loompy


def linspace_ndim(min_vals, max_vals, num_steps):  # type: ignore
	linspaces = [np.linspace(min_vals[i], max_vals[i], num=num_steps[i]) for i in range(len(min_vals))]
	m = linspaces[0]
	for i in range(1, len(linspaces)):
		m = m.repeat(len(linspaces[i]))
		m = np.vstack([m, np.tile(linspaces[i], len(m) // len(linspaces[i]))])
	return m.T


class VelocityEmbedding:
	"""
	Methods for projecting RNA velocities from expression spce or HPF latent space, to a low-dimensional embedding.
	"""
	def __init__(self, data_source: str = "expected", velocity_source: str = "velocity", genes: Union[np.ndarray, str] = None, embedding_name: str = "TSNE", neighborhood_type: Union[float, str] = "nearest", neighborhood_size: Union[int, float] = 50, points_kind: str = "cells", num_points: Union[int, Tuple[int, ...]] = None, min_neighbors: int = 2, regression_type: str = "ols") -> None:
		"""
		Create an instance of the class with the given parameters

		Args:
			data_source			Name of a layer or column attribute to use as the source data space (e.g. "expected" or "HPF")
			velocity_source		Name of a layer or column attribute to use as the source velocity (e.g. "velocity" or "HPF_Velocity")
			genes				Boolean array indicating selected genes to be used, or string giving an integer (0 or 1) row attribute (e.g. "Selected"), or None to use all genes
			embedding_name		Name of a column attribute to use as the target embedding (e.g. "TSNE" or "UMAP3D")
			neighborhood_type	Name of a column graph to use as neighborhood, or "nearest" to use nearest neighbors on the embedding, or "radius" to use all neighbors within a distance, or "voronoi" to use voronoi tesselation
			neighborhood_size	A number to use as the radius of the neighborhood on the embedding (when neighborhood == "radius"; use None to set it to 10% of range), or number of neighbors (when neighborhood_type == "nearest")
			points_kind			"cells" to use one point per cell, "grid" to use a uniform grid
			num_points			Integer, or tuple of integers giving the number of points in each embedding dimension (default: 50). If a single integer, the same number of points is used for every dimension
			min_neighbors		Minimum number of neighbors required for estimating a non-zero velocity
			regression_type		The type of regression to use when inferring velocities on the embedding: "ols", "ridge" or "lasso"
		
		Remarks:
			If data_source is the name of a layer in the Loom file, then a row attribute ds.ra.Selected == 1 can give the relevant genes; otherwise all genes are used.
		"""
		self.data_source = data_source
		self.velocity_source = velocity_source
		self.genes = genes
		self.embedding_name = embedding_name
		self.neighborhood_type = neighborhood_type
		self.neighborhood_size = neighborhood_size
		self.points_kind = points_kind
		self.num_points = num_points
		self.min_neighbors = min_neighbors
		self.regression_type = regression_type

		self.data: np.ndarray = None
		self.v_data: np.ndarray = None
		self.points: np.ndarray = None
		self.embedding: np.ndarray = None
		self.neighborhood: sparse.csr_matrix = None
		self.v_embedding: np.ndarray = None

	def fit(self, ds: loompy.LoomConnection) -> np.ndarray:
		selected: np.ndarray = None
		if isinstance(self.genes, np.ndarray):
			selected = self.genes
		elif isinstance(self.genes, str) and self.genes in ds.ra:
			selected = ds.ra[self.genes] == 1

		# Get the source data space
		if self.data_source in ds.layers:
			if selected is not None:
				self.data = ds[self.data_source][selected, :].T
			else:
				self.data = ds[self.data_source][:, :].T
		elif self.data_source in ds.ca:
			self.data = ds.ca[self.data_source]
		else:
			raise ValueError(f"Parameter 'data_source' == '{self.data_source}' must be a layer or a column attribute")

		# Get the source velocities
		if self.velocity_source in ds.layers:
			if selected is not None:
				self.v_data = ds[self.velocity_source][selected, :].T
			else:
				self.v_data = ds[self.velocity_source][:, :].T
		elif self.velocity_source in ds.ca:
			self.v_data = ds.ca[self.velocity_source]
		else:
			raise ValueError(f"Parameter 'velocity_source' == '{self.velocity_source}' must be a layer or a column attribute")
		if self.data.shape != self.v_data.shape:
			raise ValueError("Data source and velocity must be same shape")
			
		# Obtain the embedding
		if self.embedding_name in ds.ca:
			self.embedding = ds.ca[self.embedding_name]
		else:
			raise ValueError(f"Parameter 'embedding' == '{self.embedding_name}' must be a column attribute")

		# Get the points on the embedding
		if self.points_kind == "cells":
			self.points = self.embedding
			if self.neighborhood_type in ds.col_graphs:  # Check this here, because it only makes sense if points are cells
				self.neighborhood = (ds.col_graphs[self.neighborhood_type].tocsr() > 0).astype("int")
		elif self.points_kind == "grid":
			min_vals = self.embedding.min(axis=0)
			max_vals = self.embedding.max(axis=0)
			if self.num_points is None:
				num_points = np.full_like(min_vals, 50, dtype="int")
			elif isinstance(self.num_points, int):
				num_points = np.full_like(min_vals, self.num_points, dtype="int")
			elif isinstance(self.num_points, (list, tuple, np.ndarray)) and np.array(self.num_points).shape == min_vals.shape:
				num_points = np.array(self.num_points, dtype="int")
			else:
				raise ValueError("Parameter 'num_points' must be an integer, a tuple of integers (one per axis) or None")
			# Create a uniformly spaced grid
			self.points = linspace_ndim(min_vals, max_vals, num_points)

		# Get the neighborhoods
		if self.neighborhood is None:
			# Find nearest neighbors, or radius neighbors, or Voronoi tesselation
			if self.neighborhood_type == "voronoi":
				nn = NearestNeighbors().fit(self.points)
				self.neighborhood = nn.kneighbors_graph(self.embedding, n_neighbors=1).T  # Shape (n_points, n_cells)
			elif self.neighborhood_type == "nearest":
				k = int(self.neighborhood_size)
				if k < 1:
					raise ValueError("Parameter 'neighborhood_size' must be an integer >= 1 when used with 'neighborhood' == 'nearest'")
				nn = NearestNeighbors().fit(self.embedding)
				self.neighborhood = nn.kneighbors_graph(self.points, n_neighbors=k)  # Shape (n_points, n_cells)
			elif self.neighborhood_type == "radius":
				if self.neighborhood_size is None:
					radius = np.mean(max_vals - min_vals) / 10
				else:
					radius = float(self.neighborhood_size)
				if radius <= 0:
					raise ValueError("Parameter 'neighborhood_size' must be strictly positive when used with 'neighborhood' == 'radius'")
				nn = NearestNeighbors().fit(self.embedding)
				self.neighborhood = nn.radius_neighbors_graph(self.points, radius=radius)  # Shape (n_points, n_cells)
			else:
				raise ValueError("Parameter 'neighborhood_type' must be 'voronoi', 'nearest' or 'radius' when used with 'points_kind' == 'grid'")

		# Remove points that have too small neighborhoods
		nn_size = self.neighborhood.sum(axis=1).A1  # A1 converts to a vector (numpy.ndarray)
		passed = nn_size >= self.min_neighbors
		self.points = self.points[passed, :]
		self.neighborhood = self.neighborhood[passed, :]

		# At this point, we have organized all the things, as follows:
		#
		# self.data: np.ndarray of shape (n_cells, k)					the expression/component values in k-dimensional expression or latent space
		# self.v_data: np.ndarray of shape (n_cells, k)					velocities in expression/latent space
		# self.embedding: np.ndarray of shape (n_cells, d)				the embedding coordinates in d dimensions
		# self.points: np.ndarray of shape (n_points, d)				the points at which we wish to compute the embedded velocities
		# self.neighborhood: csr_matrix of shape (n_points, n_cells)	the cells that are neighbors of each point

		# Loop through all the points on the embedding
		self.v_embedding = np.zeros_like(self.points)
		n_points = self.points.shape[0]
		for i in range(n_points):
			# Compute the gradient of each of the k genes/components, using the neighbors of the current point
			neighbors = (self.neighborhood[i, :] > 0).toarray()[0]  # np.ndarray with bool elements, where the neighbors are indicated by True
			if neighbors.sum() == 0:
				continue
			neighbors_data = self.data[neighbors, :]  # Shape (n_neighbors, k)
			neighbors_embedding = self.embedding[neighbors, :]  # Shape (n_neighbors, d)
			reg = LinearRegression().fit(neighbors_embedding, neighbors_data)
			gradient = reg.coef_  # Shape (k, d)
			# Compute the average local velocity in expression/latent space
			v_binned = self.v_data[neighbors, :].mean(axis=0)  # Shape (k,)

			# Estimate v_embedding using linear regression
			if self.regression_type == "ols":
				regression = LinearRegression()
			elif self.regression_type == "ridge":
				regression = Ridge()
			elif self.regression_type == "lasso":
				regression = Lasso()
			else:
				raise ValueError("regression_type must be 'ols', 'ridge' or 'lasso'")
			self.v_embedding[i, :] = regression.fit(gradient, v_binned).coef_  # Shape (d,)

		return self.v_embedding
