import numpy as np
import scipy.sparse as sparse
from scipy.stats import linregress
from scipy.optimize import minimize_scalar
from typing import *


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
			m = minimize_scalar(lambda m: np.sum(w * (x * m - y)**2), bounds=(1e-8, up_gamma), method="bounded").x
		else:
			m = minimize_scalar(lambda m: np.sum(w * (x * m - y)**2), bounds=bounds, method="bounded").x
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


def velocity_gamma(S: np.ndarray, U: np.ndarray, limit_gamma: bool = False, bounds: Tuple[float, float] = (0, 20), maxmin_perc: List[float] = [2, 98]) -> Tuple[np.ndarray, np.ndarray]:
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