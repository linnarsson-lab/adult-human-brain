
from math import exp, lgamma, log
import logging
from typing import *
import numpy as np
from scipy.special import beta, betainc, betaln
import loompy
import numpy_groupies as npg


def expression_patterns(ds: loompy.LoomConnection, labels: np.ndarray, pep: float, f: float, cells: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Derive enrichment and trinary scores for all genes

	Args:
		ds (LoomConnection):	Dataset
		labels (numpy array):	Cluster labels (one per cell)
		pep (float):			Desired posterior error probability
		f (float):				Fraction required for a gene to be considered 'expressed'
		cells (nump array):		Indices of cells to include

	Returns:
		score1 (numpy 2d array):	Array of (n_genes, n_labels)
		score2 (numpy 2d array):	Array of (n_genes, n_labels)
		trinary (numpy 2d array):		Array of (n_genes, n_labels)

	Remarks:
		If the cells argument is provided, the labels should include only those cells. That is,
		labels.shape[0] == cells.shape[0].

		Amit says,
		regarding marker genes.
		i usually rank the genes by some kind of enrichment score.
		score1 = mean of gene within the cluster / mean of gene in all cells
		score2 = fraction of positive cells within cluster

		enrichment score = score1 * score2^power   (where power == 0.5 or 1) i usually use 1 for 10x data
	"""

	n_labels = np.max(labels) + 1

	scores1 = np.empty((ds.shape[0], n_labels))
	scores2 = np.empty((ds.shape[0], n_labels))
	trinary_pat = np.empty((ds.shape[0], n_labels))
	trinary_prob = np.empty((ds.shape[0], n_labels))

	j = 0
	for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=0):
		# vals = normalizer.normalize(vals, selection)
		for j, row in enumerate(selection):
			data = vals[j, :]
			mu0 = np.mean(data)
			f0 = np.count_nonzero(data)
			score1 = np.zeros(n_labels)
			score2 = np.zeros(n_labels)
			for lbl in range(n_labels):
				if np.sum(labels == lbl) == 0:
					continue
				sel = data[np.where(labels == lbl)[0]]
				if mu0 == 0 or f0 == 0:
					score1[lbl] = 0
					score2[lbl] = 0
				else:
					score1[lbl] = np.mean(sel) / mu0
					score2[lbl] = np.count_nonzero(sel) #  f0
			scores1[row, :] = score1
			scores2[row, :] = score2
			trinary_prob[row, :], trinary_pat[row, :] = betabinomial_trinarize_array(data, labels, pep, f)
	return (scores1, scores2, trinary_prob, trinary_pat)


def p_half(k: int, n: int, f: float) -> float:
	"""
	Return probability that at least half the cells express, if we have observed k of n cells expressing

	Args:
		k (int):	Number of observed positive cells
		n (int):	Total number of cells

	Remarks:
		Probability that at least half the cells express, when we observe k positives among n cells is:

			p|k,n = 1-(betainc(1+k, 1-k+n, 0.5)*gamma(2+n)/(gamma(1+k)*gamma(1-k+n))/beta(1+k, 1-k+n)

	Note:
		The formula was derived in Mathematica by computing

			Probability[x > f, {x \[Distributed] BetaDistribution[1 + k, 1 + n - k]}]

		and then replacing f by 0.5
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


def betabinomial_trinarize_array(array: np.ndarray, labels: np.ndarray, pep: float, f: float, n_labels: int = None) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Trinarize a vector, grouped by labels, using a beta binomial model

	Args:
		array (ndarray of ints):	The input vector of ints
		labels (ndarray of ints):	Group labels 0, 1, 2, ....
		pep (float):				The desired posterior error probability (PEP)

	Returns:
		ps (ndarray of float):		The posterior probability of expression in at least a fraction f
		expr_by_label (ndarray of float): The trinarized expression pattern (one per label)

	Remarks:
		We calculate probability p that at least half the cells express (in each group),
		and compare with pep, setting the binary pattern to 1 if p > pep,
		-1 if p < (1 - pep) and 0 otherwise.
	"""

	if n_labels is None:
		n_labels = np.max(labels) + 1
	n_by_label = np.bincount(labels, minlength=n_labels)
	k_by_label = np.zeros(n_labels)
	for lbl in range(n_labels):
		if np.sum(labels == lbl) == 0:
			continue
		k_by_label[lbl] = np.count_nonzero(array[np.where(labels == lbl)[0]])

	vfunc = np.vectorize(p_half)
	ps = vfunc(k_by_label, n_by_label, f)

	expr_by_label = np.zeros(n_labels) + 0.5
	expr_by_label[np.where(ps > (1 - pep))[0]] = 1
	expr_by_label[np.where(ps < pep)[0]] = 0

	return (ps, expr_by_label)


def save_cluster_avg(input_file: str, output_file: str) -> None:
	ds = loompy.connect(input_file)
	cells = np.where(ds.col_attrs["_Valid"] == 1)[0]
	labels = ds.col_attrs["Clusters"][cells]
	Nclust = np.max(labels) + 1
	ca = {"Cluster": np.arange(Nclust), "OriginalFile": np.array([input_file] * Nclust)}
	ra = {"Accession": ds.row_attrs["Accession"], "Gene": ds.row_attrs["Gene"]}
	m = np.empty((ds.shape[0], Nclust))
	for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=0):
		vals_avg = npg.aggregate_numba.aggregate(labels, vals, func="mean", axis=1)
		m[selection, :] = vals_avg
	loompy.create(output_file, m, ra, ca)
