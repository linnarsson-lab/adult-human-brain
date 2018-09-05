from typing import *
import numpy as np
from scipy.special import digamma, gammaln, psi
from math import lgamma
import numba


@numba.jit("float32(float64[:], float64[:])", nopython=True, cache=True)
def kullback_leibler(pk: np.ndarray, qk: np.ndarray) -> float:
	N = pk.shape[0]
	pk = pk / np.sum(pk)
	qk = qk / np.sum(qk)
	vec = np.zeros(N)
	for i in range(N):
		if pk[i] > 0 and qk[i] > 0:
			vec[i] = pk[i] * np.log(pk[i] / qk[i])
		elif pk[i] == 0 and qk[i] >= 0:
			vec[i] = 0
		else:
			vec[i] = np.inf
	S = np.sum(vec) / np.log(2)
	return S


@numba.jit("float32(float64[:], float64[:])", nopython=True, cache=True)
def jensen_shannon_divergence(pk: np.ndarray, qk: np.ndarray) -> float:
	N = pk.shape[0]
#	pk = pk / np.sum(pk)
#	qk = qk / np.sum(qk)
	m = (pk + qk) / 2

	vec = np.zeros(N)
	for i in numba.prange(N):
		if pk[i] > 0 and m[i] > 0:
			vec[i] = pk[i] * np.log(pk[i] / m[i])
		elif pk[i] == 0 and m[i] >= 0:
			vec[i] = 0
		else:
			vec[i] = np.inf
	Dpm = np.sum(vec) / np.log(2)

	vec = np.zeros(N)
	for i in numba.prange(N):
		if qk[i] > 0 and m[i] > 0:
			vec[i] = qk[i] * np.log(qk[i] / m[i])
		elif qk[i] == 0 and m[i] >= 0:
			vec[i] = 0
		else:
			vec[i] = np.inf
	Dqm = np.sum(vec) / np.log(2)

	return (Dpm + Dqm) / 2


@numba.jit("float32(float64[:], float64[:])", nopython=True, parallel=True, nogil=True)
def jensen_shannon_distance(pk: np.ndarray, qk: np.ndarray) -> float:
	"""
	Remarks:
		pk and qk must already be normalized so that np.sum(pk) == 1
	"""
	N = pk.shape[0]
#	pk = pk / np.sum(pk)
#	qk = qk / np.sum(qk)
	m = (pk + qk) / 2

	Dpm = jensen_shannon_divergence(pk, m)
	Dqm = jensen_shannon_divergence(qk, m)

	return np.sqrt((Dpm + Dqm) / 2)


def kldiv_gamma(p_shape: np.ndarray, p_rate: np.ndarray, q_shape: np.ndarray, q_rate: np.ndarray) -> float:
	"""
	Compute the Kullback-Leibler divergence between two gamma distributions.

	Remarks:
		The gamma distributions are parameterized by their shape and rate, such that the mean is shape/rate.
		See: https://en.wikipedia.org/wiki/Gamma_distribution#Kullback–Leibler_divergence

		The KL divergence is returned in units of bits (i.e. using base-2 logarithms).
	"""
	return np.sum((p_shape - q_shape) * psi(p_shape) - gammaln(p_shape) + gammaln(q_shape) + q_shape * (np.log(p_rate) - np.log(q_rate)) + p_shape * (q_rate - p_rate) / p_rate) / np.log(2)


@numba.jit("float32(float64[:], float64[:], float64, float64)", nopython=True, cache=True)
def stabilized_minkowski(x: np.ndarray, y: np.ndarray, n: float = 5000, p: float = 10) -> float:
	x_scaled = x * n / np.sum(x)
	y_scaled = y * n / np.sum(y)
	x_adj = (np.sqrt(x_scaled) + 0.8 * np.sqrt(x_scaled + 1)) / 1.8
	y_adj = (np.sqrt(y_scaled) + 0.8 * np.sqrt(y_scaled + 1)) / 1.8
	return np.sum((x_adj - y_adj) ** p) ** (1 / p)


@numba.jit("float32(float64[:], float64[:])", nopython=True, cache=True)
def minkowski10(x: np.ndarray, y: np.ndarray) -> float:
	p = 10
	return np.sum((x - y) ** p) ** (1 / p)


@numba.jit("float32(float64[:], float64[:])", nopython=True, parallel=True, nogil=True)
def multinomial_distance(p: np.ndarray, q: np.ndarray) -> float:
	N = p.shape[0]
	p_sum = p.sum()
	q_sum = q.sum()
	x = lgamma(N) + lgamma(p_sum + q_sum + N) - lgamma(p_sum + N) - lgamma(q_sum + N)
	for k in range(N):
		x += lgamma(p[k] + 1) + lgamma(q[k] + 1) - lgamma(1) - lgamma(p[k] + q[k] + 1)
	x = np.exp(x)
	return 1 - 1 / (1 + x)


@numba.jit("float32(float64[:], float64[:])", nopython=True, parallel=True, nogil=True)
def multinomial_subspace_distance(pk: np.ndarray, qk: np.ndarray) -> float:
	selected = (pk > 0) | (qk > 0)
	p = pk[selected]
	q = qk[selected]
	N = p.shape[0]
	p_sum = p.sum()
	q_sum = q.sum()
	x = lgamma(N) + lgamma(p_sum + q_sum + N) - lgamma(p_sum + N) - lgamma(q_sum + N)
	for k in range(N):
		x += lgamma(p[k] + 1) + lgamma(q[k] + 1) - lgamma(1) - lgamma(p[k] + q[k] + 1)
	x = np.exp(x)
	return 1 - 1 / (1 + x)
