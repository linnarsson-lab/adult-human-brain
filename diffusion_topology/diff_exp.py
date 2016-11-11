
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.special import beta,gamma,betainc

def is_gene_enriched(ds, cells1, cells2, gene_name):
	norm = ds.col_attrs["_TotalRNA"]
	a = np.log(ds[np.where(ds.Gene == gene), :][0, cells1]+1)/norm
	b = np.log(ds[np.where(ds.Gene == gene), :][0, cells2]+1)/norm

	(_, pval) = mannwhitneyu(a, b, alternative="greater")
	return pval

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
    return 1-(betainc(1+k, 1-k+n, 0.5)*beta(1+k,1-k+n)*gamma(2+n)/(gamma(1+k)*gamma(1-k+n)))

def betabinomial_binarize(array, labels, local_fdr):
	"""
	Binarize a vector, grouped by labels, using a beta binomial model

	Args:
		array (ndarray of ints):	The input vector of ints
		labels (ndarray of ints):	Group labels 0, 1, 2, ....
		local_fdr (float):			The desired local FDR

	Returns:
		expr_by_label (ndarray of ints): The binarized expression pattern (one per label)

	Remarks:
		We calculate probability p that at least half the cells express (in each group), and compare with local_fdr, 
		setting the binary pattern to 1 if p > local_fdr
	"""

	n_by_label = np.bincount(labels)
	k_by_label = np.zeros(len(n_by_label))
	for lbl in range(len(n_by_label)):
		k_by_label[lbl] = np.count_nonzero(array[lbl])
	
	vfunc = np.vectorize(p_half)
	ps = vfunc(k_by_label, n_by_label)

	expr_by_label = np.where(ps > local_fdr)[0]

	return expr_by_label
