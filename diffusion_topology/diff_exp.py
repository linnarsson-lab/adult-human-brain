
import numpy as np
from scipy.special import beta, gamma, betainc

def expression_patterns(ds, labels, pep, cells=None):
	"""
	Derive enrichment and trinary scores for all genes

	Args:
		ds (LoomConnection):	Dataset
		labels (numpy array):	Cluster labels (one per cell)
		cells (numpy array):	List of cells that have labels

	Returns:
		enrichment (numpy 2d array):	Array of (n_genes, n_labels)
		trinary (numpy 2d array):		Array of (n_genes, n_labels)

	Remarks:

	Amit says,
	regarding marker genes.
	i usually rank the genes by some kind of enrichment score.
	score1 = mean of gene within the cluster / mean of gene in all cells
	score2 = fraction of positive cells within cluster / fraction of positive cells in all cells

	enrichment score = score1 * score2^power   (where power == 0.5 or 1) i usually use 1 for 10x data
	"""

	n_labels = np.max(labels) + 1

	enrichment = np.empty((ds.shape[0], n_labels))
	trinary = np.empty((ds.shape[0], n_labels))
	for row in range(ds.shape[0]):
		if cells is None:
			data = ds[row, :]
		else:
			data = ds[row, :][cells]
		mu0 = np.mean(data)
		f0 = np.count_nonzero(data)
		score1 = np.empty(n_labels)
		score2 = np.empty(n_labels)
		for lbl in range(n_labels):
			if mu0 == 0:
				score1[lbl] = 0
				score2[lbl] = 0
			score1[lbl] = np.mean(data[np.where(labels == lbl)][0])/mu0
			score2[lbl] = np.count_nonzero(data[np.where(labels == lbl)])/f0
		enrichment[row, :] = score1 * score2
		trinary[row, :] = betabinomial_trinarize_array(data, labels, pep)
	return (enrichment, trinary)

def p_half(k, n):
	"""
	Return probability that at least half the cells express, if we have observed k of n cells expressing

	Args:
		k (int):	Number of observed positive cells
		n (int):	Total number of cells

	Remarks:
		Probability that at least half the cells express, when we observe k positives among n cells is:

			p|k,n = 1-(betainc(1+k, 1-k+n, 0.5)*gamma(2+n)/(gamma(1+k)*gamma(1-k+n))/beta(1+k, 1-k+n)

		We calculate this and compare with local_fdr, setting the binary pattern to 1 if p > local_fdr

	Note:
		The formula was derived in Mathematica by computing

			Probability[x > f, {x \[Distributed] BetaDistribution[1 + k, 1 + n - k]}]

		and then replacing f by 0.5
	"""
	return 1-(betainc(1+k, 1-k+n, 0.5)*beta(1+k, 1-k+n)*gamma(2+n)/(gamma(1+k)*gamma(1-k+n)))

def betabinomial_trinarize_array(array, labels, pep):
	"""
	Trinarize a vector, grouped by labels, using a beta binomial model

	Args:
		array (ndarray of ints):	The input vector of ints
		labels (ndarray of ints):	Group labels 0, 1, 2, ....
		pep (float):				The desired posterior error probability (PEP)

	Returns:
		expr_by_label (ndarray of ints): The trinarized expression pattern (one per label)

	Remarks:
		We calculate probability p that at least half the cells express (in each group),
		and compare with pep, setting the binary pattern to 1 if p > pep,
		-1 if p > (1 - pep) and 0 otherwise.
	"""

	n_labels = np.max(labels) + 1
	n_by_label = np.bincount(labels)
	k_by_label = np.zeros(n_labels)
	for lbl in range(n_labels):
		k_by_label[lbl] = np.count_nonzero(array[lbl])

	vfunc = np.vectorize(p_half)
	ps = vfunc(k_by_label, n_by_label)

	expr_by_label = np.zeros(n_labels)
	expr_by_label[np.where(ps > pep)[0]] = 1
	expr_by_label[np.where(ps > (1-pep))[0]] = -1

	return expr_by_label
