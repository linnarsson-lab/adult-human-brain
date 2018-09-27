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


class VelocityInference:
	def __init__(self, n_components: int = 10, lr: float = 0.0001, n_epochs: int = 10) -> None:
		self.lr = lr
		self.n_components = n_components

		self.n_epochs = n_epochs
		self.loss: float = 0
		self.a: np.ndarray = None
		self.b: np.ndarray = None
		self.g: np.ndarray = None

	def fit(self, ds: loompy.LoomConnection, g_init: np.ndarray = None) -> Any:
		"""
		Args:
			s      (n_genes, n_cells)
			u      (n_genes, n_cells)
		"""
		n_cells = ds.shape[1]
		logging.info("Determining selected genes")
		selected = (ds.ra.Selected == 1)
		n_genes = selected.sum()
		n_components = ds.ca.HPF.shape[1]
		logging.info("Loading spliced")
		s_data = ds["spliced"][:,:][selected, :].astype("float")
		logging.info("Loading unspliced")
		u_data = ds["unspliced"][:,:][selected, :].astype("float")
		logging.info("Loading HPF")
		m_data = ds.ra.HPF[selected, :]
		
		# Set up the optimization problem
		logging.info("Setting up the optimization problem")
		dt = torch.float
		if g_init is not None:
			g = Variable(torch.tensor(g_init, dtype=dt), requires_grad=True)
		else:
			g = Variable(0.1 * torch.ones(n_genes, dtype=dt), requires_grad=True)
		self.model = SimpleNamespace(
			s=Variable(torch.tensor(s_data, dtype=dt)),
			u=Variable(torch.tensor(u_data, dtype=dt)),
			m=Variable(torch.tensor(m_data, dtype=dt)),
			v=Variable(torch.tensor(np.random.normal(0, 1, size=(n_components, n_cells)), dtype=dt), requires_grad=True),
			b=Variable(torch.ones(n_genes, dtype=dt), requires_grad=True),
			g=g
		)
		logging.info("Optimizing")
		self.epochs(self.n_epochs)
		return self

	def epochs(self, n_epochs: int) -> Any:
		m = self.model
		loss_fn = torch.nn.MSELoss()
		optimizer = torch.optim.SGD([m.v, m.b, m.g], lr=self.lr)

		for epoch in trange(n_epochs):
			optimizer.zero_grad()
			left = m.m @ m.v
			right = m.b.unsqueeze(1) * m.u - m.g.unsqueeze(1) * m.s - m.m @ m.v
			loss_out = loss_fn(left, right)
			loss_out.backward()
			optimizer.step()
			m.b.data.clamp_(min=0)
			m.g.data.clamp_(min=0)

		self.loss = float(loss_out)
		self.v = m.v.detach().numpy()
		self.b = m.b.detach().numpy()
		self.g = m.g.detach().numpy()
		return self


def _fit1_slope_weighted(y: np.ndarray, x: np.ndarray, w: np.ndarray, limit_gamma: bool=False, bounds: Tuple[float, float]=(0, 20)) -> float:
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
	return m


def fit_gamma(S: np.ndarray, U: np.ndarray, limit_gamma: bool = False, bounds: Tuple[float, float] = (0, 20), maxmin_perc: List[float] = [2, 98]) -> np.ndarray:
	"""Loop through the genes and fits the slope
	S: np.ndarray, shape=(genes, cells)
		the independent variable (spliced)
	U: np.ndarray, shape=(genes, cells)
		the dependent variable (unspliced)

	Remarks:
		Original code for velocyto by Gioele La Manno. This simplified version
		by Sten Linnarsson. http://velocyto.org
	"""

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

	slopes = np.zeros(S.shape[0], dtype="float32")
	for i in range(S.shape[0]):
		m = _fit1_slope_weighted(U[i, :], S[i, :], W[i, :], limit_gamma)
		slopes[i] = m
	return slopes


# with loompy.connect("/Users/stelin/dh_20170213/L1_Subcortex.loom") as ds:
# 	s = ds["spliced"][ds.ra.Selected == 1, :]
# 	u = ds["unspliced"][ds.ra.Selected == 1, :]
# 	gammas = fit_slope_weighted(s,u)