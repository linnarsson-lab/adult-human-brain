from math import exp, lgamma, log
import numpy as np
from scipy.special import betainc, betaln
import loompy


class Trinarizer:
	"""
	Compute trinarization probability per cluster
	"""
	def __init__(self, f: float = 0.2) -> None:
		self.f = f
		self.p_half_vectorized = np.vectorize(self.p_half)

	def fit(self, ds: loompy.LoomConnection) -> np.ndarray:

		#  Get number of observed positive cells per gene per cluster
		k = ds["nonzeros"][:, :].T
		# Get number of cells
		n = ds.ca.NCells

		p = self.p_half_vectorized(k.T.astype("float32"), n.astype("float32"), self.f).astype("float32")
		return p

	def p_half(self, k: int, n: int, f: float) -> float:
		"""
		Return probability that at least half the cells express, if we have observed k of n cells expressing

		Args:
			k (int):	Number of observed positive cells
			n (int):	Total number of cells

		Remarks:
			Probability that at least a fraction f of the cells express, when we observe k positives among n cells is:

				p|k,n = 1-(betainc(1+k, 1-k+n, f)*gamma(2+n)/(gamma(1+k)*gamma(1-k+n))/beta(1+k, 1-k+n)

		Note:
			The formula was derived in Mathematica by computing

				Probability[x > f, {x \[Distributed] BetaDistribution[1 + k, 1 + n - k]}]
		"""

		# These are the prior hyperparameters beta(a,b)
		a = 1.5
		b = 2

		# We really want to calculate this:
		# p = 1-(betainc(a+k, b-k+n, 0.5)*beta(a+k, b-k+n)*gamma(a+b+n)/(gamma(a+k)*gamma(b-k+n)))
		#
		# But it's numerically unstable, so we need to work on log scale (and special-case the incomplete beta)

		incb = betainc(a + k, b - k + n, f)
		if incb == 0:
			p = 1.0
		else:
			p = 1.0 - exp(log(incb) + betaln(a + k, b - k + n) + lgamma(a + b + n) - lgamma(a + k) - lgamma(b - k + n))
		return p
