
import numpy as np
from copy import copy
import logging
from typing import *
from numpy_groupies import aggregate_numba as agg
from tqdm import trange, tqdm

# TODO: Bregman divergence

# px = x./(x + r); % negbin parameter for x
# py = y./(y + r); % negbin parameter for y

# bxy = x.*(log(px)-log(py)) + r*(log(1-px)-log(1-py));

# Note this is undefined if x or y=0, so you really need to compute image011.png, where image012.png is a
# regularization factor (0.1 seems to work well). The parameter r measures the amount of variability,
# 2 seems to work well for your data.


class Facet:
	def __init__(self, name: str, k: int=2, max_k: int=5, n_genes: int=100, genes: List[int]=[], adaptive: bool=False) -> None:
		"""
		Create a Facet object

		Args:
			k (int):				Number of (initial) clusters in this facet
			max_k (int):			Maximum number of clusters in this facet
			n (int): 				Number of genes to allocate to this Facet
			genes (List[string]):	Genes to use to initialize this Facet
			adaptive (bool):		If true, the number of clusters is increased until BIC is minimized
		"""
		self.k = k
		self.max_k = max_k
		self.n_genes = n_genes
		self.name = name
		self.genes = genes
		self.adaptive = adaptive

		# fields used during fitting
		self.labels = None  # type: np.ndarray
		self.S = None  # type: np.ndarray
		self.pi_k = None  # type: np.ndarray
		self.y_g = None  # type: np.ndarray
		self.BIC = None  # type: np.ndarray


class FacetLearning:
	def __init__(self, facets: List[Facet], r: float = 2.0, alpha: float = 1.0, max_iter: int = 100, gene_names: List[str]=None) -> None:
		"""
		Create a FacetLearning object

		Args:
			facets (List[Facet]):	The facet definitions
			r (float):				The overdispersion
			alpha (float):			The regularization factor
			max_iter (int):			The number of EM iterations to run
		"""
		self.facets = facets
		self.r = r
		self.alpha = alpha
		self.max_iter = max_iter
		self.n_split_tries = 5
		self.min_cluster_size = 10
		self.gene_names = gene_names

	def fit_predict(self, X: np.ndarray) -> np.ndarray:
		n_cells = X.shape[0]
		for facet in self.facets:
			if facet.labels is None:
				facet.labels = np.random.randint(facet.k, size=X.shape[0])
			facet.S = np.random.choice(X.shape[1], size=facet.n_genes, replace=False)
			if len(facet.genes) > 0:
				facet.S[:len(facet.genes)] = facet.genes
			facet.pi_k = (np.bincount(facet.labels, minlength=facet.k) + 1) / (n_cells + facet.k)
			np.ones(facet.k) / facet.k

		# Keep alternating EM and splitting until no more splits
		splitting = True
		n_cycles = 0
		last_BIC = None  # type: float
		while splitting:
			# Run EM on the combined facets
			for _ in trange(self.max_iter, desc="EM", leave=False):
				self._E_step(X)
				self._M_step(X)
			splitting = False
			for facet in self.facets:
				if self.gene_names is not None:
					logging.info(facet.name + " " + str(self.gene_names[facet.S]))
				if facet.adaptive:
					if last_BIC is not None and facet.BIC > last_BIC:
						logging.info("Stopped splitting because BIC did not drop")
						facet.adaptive = False
						continue
					n_cycles += 1
					last_BIC = facet.BIC
					logging.info("Cycle %d, splitting facet '%s' with %d clusters %s", n_cycles, facet.name, facet.k, str(np.bincount(facet.labels, minlength=facet.k)))
					logging.info("BIC: %f", facet.BIC)
					if self._split_facet(X, facet) > 0:
						splitting = True
		labels = []
		for facet in self.facets:
			labels.append(facet.labels)
		return np.array(labels).T

	def _E_step(self, X: np.ndarray) -> None:
		for facet in self.facets:
			X_S = X[:, facet.S]
			n_cells = X.shape[0]
			# (n_cells, k)
			z_ck = np.zeros((n_cells, facet.k))
			# (k, n_S)
			mu_gk = agg.aggregate(facet.labels, X_S, func='mean', fill_value=0, size=facet.k, axis=0) + 0.01
			# (k, n_S)
			p_gk = mu_gk / (mu_gk + self.r)
			# (n_cells, k)
			# z_ck += X_S.dot((np.log(p_gk) + self.r*np.log(1-p_gk)).transpose())
			z_ck += np.log(facet.pi_k)
			z_ck += np.log(p_gk).dot(X_S.transpose()).transpose()
			z_ck += np.sum(self.r * np.log(1 - p_gk), axis=1)
			# (n_cells)
			facet.labels = np.argmax(z_ck, axis=1)
			# Add 1 to each as a pseudocount to avoid zeros
			facet.pi_k = (np.bincount(facet.labels, minlength=facet.k) + 1) / (n_cells + facet.k)

	def _M_step(self, X: np.ndarray) -> None:
		n_genes = X.shape[1]
		n_cells = X.shape[0]

		all_yg = np.zeros((n_genes, len(self.facets)))
		for i, facet in enumerate(self.facets):
			# (n_genes)
			facet.y_g = np.zeros(n_genes)
			facet_L_g = np.zeros(n_genes)
			# (k, n_genes)
			mu_gk = agg.aggregate(facet.labels, X, func='mean', fill_value=0, size=facet.k, axis=0) + 0.01
			# (k, n_genes)
			p_gk = mu_gk / (mu_gk + self.r)
			# (n_genes)
			mu_g0 = X.mean(axis=0) + 0.01
			# (n_genes)
			p_g0 = mu_g0 / (mu_g0 + self.r)
			for c in range(n_cells):
				p_gkc = p_gk[facet.labels[c], :]
				facet.y_g += X[c, :] * (np.log(p_gkc) - np.log(p_g0)) + self.r * np.log(1 - p_gkc) - np.log(1 - p_g0)
				facet_L_g += X[c, :] * np.log(p_gkc) + self.r * np.log(1 - p_gkc)
			all_yg[:, i] = facet.y_g
			facet_L = np.sum(facet_L_g)
			# BIC = -2 log(L) + (n_selected_genes * k + k + n_genes) * log(n_cells)
			facet.BIC = -2 * facet_L + (facet.n_genes * facet.k + facet.k + n_genes) * np.log(n_cells)

		# Compute the regularized likelihood gains
		all_yg_sum = np.sum(all_yg, axis=1)
		all_yg_regularized = 2 * all_yg - self.alpha * all_yg_sum[:, None]

		for i, facet in enumerate(self.facets):
			if len(facet.genes) > 0:
				if len(facet.genes) < facet.n_genes:
					facet.S = np.argsort(all_yg_regularized[:, i], axis=0)[-facet.n_genes:]
				facet.S[:len(facet.genes)] = facet.genes
			else:
				facet.S = np.argsort(all_yg_regularized[:, i], axis=0)[-facet.n_genes:]

	def _suggest_splits(self, X: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Calculate maximum likelihood gain for each gene when cells are split in two

		Args:
			X:			The data matrix (n_cells, n_genes)
			cells:		The cells to consider

		Returns:
			gains:		The likelihood gain for each gene (n_genes)
			thetas:		The optimal split point for each gene (n_genes)

		Remarks:
			This code closely follows the original MATLAB code by Kenneth Harris
		"""
		logging.debug("Calculating optimal splits for %d cells", cells.shape[0])
		n_cells = cells.shape[0]
		xs = np.sort(X[cells, :], axis=0)

		# cumulative sums for top and bottom halves of expression
		# - to evaluate splitting each gene in each position
		# (n_cells, n_genes)
		cx1 = np.cumsum(xs, axis=0)
		cx2 = cx1[::-1, :] - cx1

		# mean expression for top and bottom halves
		# n1 = 1..n_cells
		n1 = np.arange(n_cells) + 1
		n2 = n_cells - n1
		regN = 1e-4
		regD = 1
		# (n_cells, n_genes)
		mu1 = (cx1 + regN) / (n1 + regD)[:, None]
		mu2 = (cx2 + regN) / (n2 + regD)[:, None]

		# nbin parameters (n_cells, n_genes)
		p1 = mu1 / (mu1 + self.r)
		p2 = mu2 / (mu2 + self.r)

		L1 = cx1 * np.log(p1) + self.r * (np.log(1 - p1) * n1[:, None])
		L2 = cx2 * np.log(p2) + self.r * (np.log(1 - p2) * n2[:, None])

		dL = (L1 + L2) - L1[::-1, :]
		split_pos = np.argmax(dL, axis=0)
		gains = dL.T[range(len(split_pos)), split_pos].T
		thetas = xs.T[range(len(split_pos)), split_pos].T
		return (gains, thetas)

	def _evaluate_splits(self, X: np.ndarray, facet: Facet, cells: np.ndarray, genes: np.ndarray, thetas: np.ndarray) -> Tuple[np.ndarray, int, float]:
		"""
		Evaluate how well it works to split this particular cluster by each of the given genes

		Args:
			X:				The data matrix (n_cells, n_genes)
			facet:			The facet to work with
			cells:			The cells to split
			genes:			The genes to consider
			thetas:			The values to split by

		Returns:
			best_classes:	The classes of the best split (0s and 1s only)
			best_gene:		The best gene for splitting, or -1 if no improvement
			delta_BIC:		The delta-BIC score of the best gene
		"""
		n_cells = cells.shape[0]
		x = X[cells, :]

		# First run a one-step EM with k == 1 to force calculation of BIC
		f0 = copy(facet)
		f0.k = 1
		f0.adaptive = False
		f0.labels = np.zeros(cells.shape[0], dtype='int')
		fl = FacetLearning([f0], self.r, self.alpha, max_iter=1)
		_ = fl.fit_transform(x)
		original_BIC = f0.BIC
		best_BIC = original_BIC
		best_labels = f0.labels
		best_gene = -1
		best_i = -1
		for i, gene in enumerate(genes):
			theta = thetas[i]
			f0 = copy(facet)
			f0.k = 2
			f0.adaptive = False
			f0.labels = (x[:, gene] > theta).astype('int')
			fl = FacetLearning([f0], self.r, self.alpha, max_iter=20)
			labels = fl.fit_transform(x)
			if f0.BIC < best_BIC:
				best_BIC = f0.BIC
				best_labels = f0.labels
				best_gene = gene
				best_i = i

		if best_BIC >= original_BIC:
			logging.debug("No improvement over original BIC score")
		else:
			logging.debug("Splitting %d cells into %s would reduce BIC by %f", n_cells, str(np.bincount(best_labels, minlength=2)), best_BIC - original_BIC)
		logging.info("Best gene: " + self.gene_names[best_gene])

		return (best_labels, best_gene, best_BIC - original_BIC)

	def _split_facet(self, X: np.ndarray, facet: Facet) -> int:
		"""
		Split one cluster of the facet, except if BIC fails to drop

		Args:
			X:			Expression matrix
			facet:		The facet to split

		Returns:
			n_split:	Number of clusters that were split
		"""
		n_cells = X.shape[0]
		did_split = 0
		best_split_labels = None  # type: np.ndarray
		best_split_cells = None  # type: np.ndarray
		best_delta_BIC = None  # type: float
		best_k = None  # type: int
		for k in range(facet.k):
			cells = np.where(facet.labels == k)[0]
			if cells.shape[0] > self.min_cluster_size:
				(gains, thetas) = self._suggest_splits(X, cells)
				best_genes = np.argsort(gains)[-self.n_split_tries:]
				best_thetas = thetas[-self.n_split_tries:]
				(split_labels, split_gene, delta_BIC) = self._evaluate_splits(X, facet, cells, best_genes, best_thetas)
				if split_gene != -1 and (best_delta_BIC is None or delta_BIC < best_delta_BIC):
					did_split = True
					best_delta_BIC = delta_BIC
					best_split_labels = split_labels
					best_split_cells = cells
					best_k = k
		if did_split:
			facet.k += 1
			# split_labels is a list of 0 and 1
			# transform it into a list of best_k and (facet.k - 1)
			facet.labels[best_split_cells] = best_split_labels * (facet.k - 1 - best_k) + best_k
			facet.pi_k = (np.bincount(facet.labels, minlength=facet.k) + 1) / (n_cells + facet.k)
			logging.info("Decided to split into %d clusters %s", facet.k, str(np.bincount(facet.labels, minlength=facet.k)))
			return 1
		else:
			return 0

