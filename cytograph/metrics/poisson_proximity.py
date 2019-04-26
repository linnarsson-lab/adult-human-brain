import math

import numba
import numpy as np
from scipy.special import gammaln


class PoissonProximity:
	def __init__(self, A: float, B: float) -> None:
		self.A = A
		self.B = B

	def logF(self, x: np.ndarray, y: np.ndarray) -> float:
		A = self.A
		B = self.B
		N = x.shape[0]

		x_norm = np.sum(x)
		y_norm = np.sum(y)
		C = y_norm / x_norm
		BC = B * C

		F = 0
		F += N * gammaln(A)
		F += - N * A * np.log(B + BC + 1)
		F += x_norm * np.log(C)
		F += (x_norm + y_norm) * np.log(B / (B + BC + 1))
		F += gammaln(A + x + y).sum()
		F += N * A * np.log((B + 1) * (BC + 1))
		F += - x_norm * np.log(BC / (BC + 1))
		F += - y_norm * np.log(B / (B + 1))
		F += - gammaln(A + x).sum()
		F += - gammaln(A + y).sum()
		return F

	@numba.jit(nopython=True, cache=True)
	def logF_jitted(self, x: np.ndarray, y: np.ndarray) -> float:
		A = self.A
		B = self.B
		x_norm = np.sum(x)
		y_norm = np.sum(y)
		C = y_norm / x_norm
		
		x_norm = np.sum(x)
		y_norm = np.sum(y)

		F = 0
		F += x_norm * np.log(C)
		F += (x_norm + y_norm) * np.log(B / (B + B * C + 1))

		temp = A + x + y
		for ix in numba.prange(temp.shape[0]):
			temp[ix] = math.lgamma(temp)
		F += temp.sum()

		F += x_norm * np.log(B * C / (B * C + 1))

		temp = A + x
		for ix in numba.prange(temp.shape[0]):
			temp[ix] = math.lgamma(temp)
		F += temp.sum()
		
		temp = B + x
		for ix in numba.prange(temp.shape[0]):
			temp[ix] = math.lgamma(temp)
		F += temp.sum()

		return F
