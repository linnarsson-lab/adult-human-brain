import numpy as np
import torch
from torch.autograd import Variable
from sklearn.decomposition import PCA
import scipy.sparse as sparse
from tqdm import trange
from scipy.stats import linregress
from typing import *
from types import SimpleNamespace
import loompy
import logging
import scipy

#
# Idea
#
# Set up an optimization problem
#
# Constants:
#
#		s		spliced
#		u		unspliced
#
# Variables:
#
#		dT		for each neighbor pair
#		gamma	for each gene
#
# Probability:
#
#		Poisson(s[j], lambda=s[i] + v[i] * dT[i, j])
#		v = u - gamma s

class VelocityInference:
	def __init__(self, lr: float = 0.0001, n_epochs: int = 1000) -> None:
		self.lr = lr

		self.n_epochs = n_epochs
		self.loss: float = 0
		self.a: np.ndarray = None
		self.b: np.ndarray = None
		self.g: np.ndarray = None
		self.gamma: np.ndarray = None
		self.weights: np.ndarray = None

	def fit_loom(self, ds: loompy.LoomConnection) -> Any:
		genes = ds.ra.Selected == 1
		n_genes = genes.sum()
		n_cells = ds.shape[1]
		logging.info("Loading spliced and unspliced data for selected genes")
		s = np.empty((n_genes, n_cells))
		u = np.empty((n_genes, n_cells))
		for (ix, selected, view) in ds.scan(axis=1, layers=["spliced_exp", "unspliced_exp"]):
			s[:, selected] = view.layers["spliced_exp"][genes, :]
			u[:, selected] = view.layers["unspliced_exp"][genes, :]
		logging.info("Loading HPF beta")
		beta = ds.ra.HPF_beta[genes, :]
		logging.info("Fitting gamma")
		self.gamma, self.weights = fit_gamma(s, u, True)
		return self.fit(s, u, beta)

	def fit(self, s_data: np.ndarray, u_data: np.ndarray, hpf_beta: np.ndarray) -> Any:
		"""
		Args:
			s_data		(n_genes, n_cells)
			u_data		(n_genes, n_cells)
			hpf_beta	(n_genes, n_factors)
		"""
		n_genes, n_cells = s_data.shape
		n_components = hpf_beta.shape[1]

		if self.gamma is None:
			logging.info("Initializing gamma by quantile fit")
			self.gamma, self.weights = fit_gamma(s_data, u_data, True)

		# Set up the optimization problem
		logging.info("Setting up the optimization problem")
		dt = torch.float
		self.model = SimpleNamespace(
			s=Variable(torch.tensor(s_data, dtype=dt), requires_grad=True),
			u=Variable(torch.tensor(u_data, dtype=dt), requires_grad=True),
			w=Variable(torch.tensor(self.weights, dtype=dt), requires_grad=True),
			hpf=Variable(torch.tensor(hpf_beta, dtype=dt), requires_grad=True),
			v=Variable(torch.tensor(np.random.normal(0, 1, size=(n_components, n_cells)), dtype=dt), requires_grad=True),
			b=Variable(torch.ones(n_genes, dtype=dt), requires_grad=True),
			g=Variable(torch.tensor(self.gamma, dtype=dt), requires_grad=True)
		)
		logging.info("Optimizing")
		self.epochs(self.n_epochs)
		return self

	def epochs(self, n_epochs: int) -> Any:
		m = self.model
		optimizer = torch.optim.SGD([m.v, m.g, m.b], lr=self.lr)
		t = trange(n_epochs)

		for epoch in t:
			left = m.hpf @ m.v
			right = m.b.unsqueeze(1) * m.u - m.g.unsqueeze(1) * m.s
			loss = torch.mean(m.w * (right - left) ** 2)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			m.b.data.clamp_(min=0)
			m.g.data.clamp_(min=0)
			t.set_description(f"MSE={loss}")
			t.refresh()

		self.loss = float(loss)
		self.v = m.v.detach().numpy()
		self.b = m.b.detach().numpy()
		self.g = m.g.detach().numpy()
		return self


def _fit1_slope_weighted(y: np.ndarray, x: np.ndarray, w: np.ndarray, limit_gamma: bool = False, bounds: Tuple[float, float] = (0, 20)) -> float:
	"""Simple function that fit a weighted linear regression model without intercept
	"""
	if not np.any(x):
		m = np.NAN
	elif not np.any(y):
		m = 0
	else:
		if limit_gamma:
			if np.median(y) > np.median(x):
				high_x = x > np.percentile(x, 90)
				up_gamma = np.percentile(y[high_x], 10) / np.median(x[high_x])
				up_gamma = np.maximum(1.5, up_gamma)
			else:
				up_gamma = 1.5  # Just a bit more than 1
			m = scipy.optimize.minimize_scalar(lambda m: np.sum(w * (x * m - y)**2), bounds=(1e-8, up_gamma), method="bounded").x
		else:
			m = scipy.optimize.minimize_scalar(lambda m: np.sum(w * (x * m - y)**2), bounds=bounds, method="bounded").x
	if not np.isfinite(m):
		return 1
	return m


def quantile_weights(S: np.ndarray, U: np.ndarray) -> np.ndarray:
	"""
	Find weights suitable for a quantile fit of gamma

	Args:
		S: np.ndarray, shape=(n_genes, n_cells), the independent variable (spliced)
		U: np.ndarray, shape=(n_genes, n_cells), the dependent variable (unspliced)
	
	Returns:
		W: np.ndarray, shape=(n_genes, n_cells), the weights
	"""
	maxmin_perc = [2, 98]

	denom_S = np.percentile(S, 99.9, 1)
	if np.sum(denom_S == 0):
		denom_S[denom_S == 0] = np.maximum(np.max(S[denom_S == 0, :], 1), 0.001)
	denom_U = np.percentile(U, 99.9, 1)
	if np.sum(denom_U == 0):
		denom_U[denom_U == 0] = np.maximum(np.max(U[denom_U == 0, :], 1), 0.001)
	S_maxnorm = S / denom_S[:, None]
	U_maxnorm = U / denom_U[:, None]
	X = S_maxnorm + U_maxnorm
	down, up = np.percentile(X, maxmin_perc, axis=1)
	W = ((X <= down[:, None]) | (X >= up[:, None])).astype(float)
	return W


def fit_gamma(S: np.ndarray, U: np.ndarray, limit_gamma: bool = False, bounds: Tuple[float, float] = (0, 20), maxmin_perc: List[float] = [2, 98]) -> Tuple[np.ndarray, np.ndarray]:
	"""Loop through the genes and fits the slope
	S: np.ndarray, shape=(genes, cells)
		the independent variable (spliced)
	U: np.ndarray, shape=(genes, cells)
		the dependent variable (unspliced)

	Remarks:
		Original code for velocyto by Gioele La Manno. This simplified version
		by Sten Linnarsson. http://velocyto.org
	"""
	W = quantile_weights(S, U)

	slopes = np.zeros(S.shape[0], dtype="float32")
	for i in range(S.shape[0]):
		m = _fit1_slope_weighted(U[i, :], S[i, :], W[i, :], limit_gamma)
		slopes[i] = m
	return slopes, W
