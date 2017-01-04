
import numpy as np
import logging
from numpy_groupies import aggregate_numba as agg


class ProMMT(object):
	def __init__(self, n_S, k=2, r=2, max_iter=1000, pep=0.05, f=0.2, min_discordant=5):
		self.r = r
		self.k = k
		self.max_iter = max_iter
		self.pi_k = np.ones(k)/k
		self.n_S = n_S
		self.pep = pep
		self.f = f
		self.min_discordant = min_discordant

	def fit_predict(self, X, genes=None) -> np.ndarray:
		self.labels = np.random.randint(self.k, size=X.shape[0])

		if genes is not None:
			self.S = genes
		else:
			self.S = np.random.choice(X.shape[1], size=self.n_S, replace=False)

		for _ in range(self.max_iter):
			self._E_step(X)
			self._M_step(X)
		logging.info("Log likelihood: " + str(self.L))
		logging.info("BIC: " + str(self.BIC))
		return self.labels

	def _E_step(self, X):
		X_S = X[:, self.S]
		n_cells = X.shape[0]
		# (n_cells, k)
		z_ck = np.zeros((n_cells, self.k))
		# (k, n_S)
		mu_gk = agg.aggregate(self.labels, X_S, func='mean', fill_value=0, size=self.k, axis=0) + 0.01
		# (k, n_S)
		p_gk = mu_gk / (mu_gk + self.r)
		# (n_cells, k)
		#z_ck += X_S.dot((np.log(p_gk) + self.r*np.log(1-p_gk)).transpose())
		z_ck += np.log(self.pi_k)
		z_ck += np.log(p_gk).dot(X_S.transpose()).transpose()
		z_ck += np.sum(self.r*np.log(1-p_gk), axis=1)
		# (n_cells)
		self.labels = np.argmax(z_ck, axis=1)

	def _M_step(self, X):
		n_genes = X.shape[1]
		n_cells = X.shape[0]
		# (n_genes)
		y_g = np.zeros(n_genes)
		# (n_genes)
		L_g = np.zeros(n_genes)
		# (k, n_genes)
		mu_gk = agg.aggregate(self.labels, X, func='mean', fill_value=0, size=self.k, axis=0) + 0.01
		# (k, n_genes)
		p_gk = mu_gk / (mu_gk + self.r)
		# (n_genes)
		mu_g0 = X.mean(axis=0) + 0.01
		# (n_genes)
		p_g0 = mu_g0 / (mu_g0 + self.r)
		for c in range(n_cells):
			p_gkc = p_gk[self.labels[c], :]
			y_g += X[c, :]*(np.log(p_gkc)-np.log(p_g0))
			y_g += self.r*np.log(1 - p_gkc)-np.log(1 - p_g0)
			L_g += X[c, :]*np.log(p_gkc) + self.r*np.log(1 - p_gkc)
		self.S = np.argsort(y_g, axis=0)[-self.n_S:]
		self.L = np.sum(L_g)
		self.BIC = -2*self.L + (self.n_S*self.k + self.n_S + self.k)*np.log(n_cells)
		self.y_g = y_g
		# Add 1 to each as a pseudocount to avoid zeros
		self.pi_k = (np.bincount(self.labels, minlength=self.k) + 1)/(X.shape[0] + self.k)


# Deal with differences in cell size (in p_gk)
# Select S/k genes per group


## Bregman divergence

# px = x./(x + r); % negbin parameter for x
# py = y./(y + r); % negbin parameter for y
 
# bxy = x.*(log(px)-log(py)) + r*(log(1-px)-log(1-py));
 
# Note this is undefined if x or y=0, so you really need to compute image011.png, where image012.png is a regularization factor (0.1 seems to work well). The parameter r measures the amount of variability, 2 seems to work well for your data.
 