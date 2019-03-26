import numpy as np
import scipy.sparse as sparse
from sklearn.neighbors import BallTree, NearestNeighbors
import logging
import loompy
import cytograph as cg
import logging
from scipy.special import gammaln
from scipy.spatial.distance import minkowski
from math import lgamma
import math
import numba
import tempfile
from subprocess import Popen
from pathlib import Path
from typing import *



@numba.jit("float32(float64[:], float64[:])", nopython=True, cache=True)
def poisson_distance_model_selection(x: np.ndarray, y: np.ndarray) -> float:
	"""
	Calculate the (log of the) Bayes factor for a model selection between M1 (single product-Poisson) ansd M2 (two product-Poissons)

	Args:
		x, y		Input vectors
	"""
	A = 1
	B = 5
	N = x.shape[0]

	x_norm = np.sum(x)
	y_norm = np.sum(y)
	C = y_norm / x_norm
	BC = B * C

	F = 0
	F += N * lgamma(A)
	F += - N * A * np.log(B + BC + 1)
	F += x_norm * np.log(C)
	F += (x_norm + y_norm) * np.log(B / (B + BC + 1))

	temp = A + x + y
	for ix in numba.prange(temp.shape[0]):
		temp[ix] = lgamma(temp[ix])
	F += temp.sum()

	F += N * A * np.log((B + 1) * (BC + 1))
	F += - x_norm * np.log(BC / (BC + 1))
	F += - y_norm * np.log(B / (B + 1))

	temp = A + x
	for ix in numba.prange(temp.shape[0]):
		temp[ix] = math.lgamma(temp[ix])
	F += - temp.sum()
	
	temp = B + x
	for ix in numba.prange(temp.shape[0]):
		temp[ix] = math.lgamma(temp[ix])
	F += - temp.sum()

	return -F


@numba.jit("float32(float64[:], float64[:])", nopython=True, cache=True)
def poisson_distance_cosine_expectation(x: np.ndarray, y: np.ndarray) -> float:
	"""
	Calculate the weighted cosine similarity between two vectors living in (partially overlapping) subspaces,
	normalized by the local expected cosine distance of a Poisson-distributed sample around x (in the subspace)

	Args:
		x, y		Input vectors, with positive values indicating the active subspace
	"""
	subspace = ((x >= 0) & (y >= 0))  # .astype('int32')
	a = subspace * x
	b = subspace * y
	a_norm = np.linalg.norm(a)
	b_norm = np.linalg.norm(b)
	if a_norm == 0 or b_norm == 0:
		return 1
	cosd = 1 - np.dot(a, b) / (a_norm * b_norm)
	ssq = np.power(a, 2).sum()
	expected_cosd = 2 - 2 * ssq / (np.sqrt(ssq + a.sum()) * a_norm)
	scaled_cosd = cosd / expected_cosd
	return scaled_cosd * subspace.sum() / x.shape[0]


class AbsoluteManifold:
	def __init__(self, radius: float = 1, metric: str = "model") -> None:
		self.metric = metric
		self.radius = radius

	def fit(self, data: np.ndarray) -> Any:
		self.data = data
		self.n_samples = data.shape[0]
		if self.metric == "model":
			metric = poisson_distance_model_selection
		elif self.metric == "cosine_exp":
			metric = poisson_distance_cosine_expectation
		elif self.metric == "variance_stabilized":
			metric = cg.stabilized_minkowski

		nn = NearestNeighbors(metric=metric, radius=self.radius, algorithm="ball_tree")
		nn.fit(self.data)
		self.rnn = nn.radius_neighbors_graph(self.data, radius=self.radius)
		self.rnn.setdiag(1)
		self.n_neighbors = self.rnn.sum(axis=1).A1
		return self

	def sample(self, N: int, pseudocounts: int = 0) -> np.ndarray:
		"""
		Return a sample from the manifold with uniform density distribution
		Args:
			N				Number of samples to return
		Returns:
			Array of indexes of the selected samples
		"""
		p = cg.div0(1, cg.div0(self.n_neighbors + pseudocounts, self.rnn.nnz + pseudocounts * self.n_samples))
		p = p / p.sum()
		result: Set[int] = set()
		for ix in range(N):
			temp = np.random.choice(np.arange(self.n_samples), p=p, replace=False)
			while temp in result:
				temp = np.random.choice(np.arange(self.n_samples), p=p, replace=False)
			result.add(temp)
		return np.sort(np.array(list(result)))

	def outliers(self, min_neighbors: int = 1) -> np.ndarray:
		return self.n_neighbors <= min_neighbors

	def smoothen(self, normalize: bool = False) -> np.ndarray:
		if not normalize:
			return self.rnn.dot(self.data)

		# Compute size-corrected Poisson MLE rates
		size_factors = self.data.sum(axis=1)
		data_std = (self.data.T / size_factors).T * np.median(size_factors)
		# Average of MLE rates for neighbors
		return (self.rnn.dot(data_std).T / self.n_neighbors).T
