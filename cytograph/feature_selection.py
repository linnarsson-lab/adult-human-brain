import logging
import numpy as np
from typing import *
from sklearn.svm import SVR
import loompy


class FeatureSelection:
	def __init__(self, n_genes: int) -> None:
		self.n_genes = n_genes
		self.genes = None  # type: np.ndarray
		self.mu = None  # type: np.ndarray
		self.sd = None  # type: np.ndarray
		self.totals = None  # type: np.ndarray

	def fit(self, ds: loompy.LoomConnection, cells: np.ndarray = None, mu: np.ndarray = None, sd: np.ndarray = None, mask: np.ndarray=None) -> np.ndarray:
		"""
		Fits a noise model (CV vs mean)

		Args:
			ds (LoomConnection):	Dataset
			n_genes (int):	number of genes to include
			cells (ndarray): cells to include when computing mean and CV (or None)
			mu, std: 		Precomputed mean and standard deviations (optional)

		Returns:
			ndarray of selected genes (list of ints)
		"""
		if mu is None or sd is None:
			(mu, sd) = ds.map((np.mean, np.std), axis=0, selection=cells)

		if "_Valid" in ds.ra:
			valid = ds.ra._Valid == 1
		else:
			valid = np.ones(ds.shape[0], dtype='bool')
		if mask is not None:
			valid = np.logical_and(valid, np.logical_not(mask))
		valid = np.logical_and(valid, ds.row_attrs["Gene"] != "Xist")
		valid = np.logical_and(valid, ds.row_attrs["Gene"] != "Tsix")
		valid = valid.astype('int')

		ok = np.logical_and(mu > 0, sd > 0)
		cv = sd[ok] / mu[ok]
		log2_m = np.log2(mu[ok])
		log2_cv = np.log2(cv)

		svr_gamma = 1000. / len(mu[ok])
		clf = SVR(gamma=svr_gamma)
		clf.fit(log2_m[:, np.newaxis], log2_cv)
		fitted_fun = clf.predict
		# Score is the relative position with respect of the fitted curve
		score = log2_cv - fitted_fun(log2_m[:, np.newaxis])
		score = score * valid[ok]
		self.genes = np.where(ok)[0][np.argsort(score)][-self.n_genes:]

		return self.genes


# This works on a matrix in memory
def feature_selection(data: np.ndarray, n_genes: int) -> np.ndarray:
	valid = (data.sum(axis=0) > 0).astype('int')
	mu = np.mean(data, axis=0)
	sd = np.std(data, axis=0)
	ok = np.logical_and(mu > 0, sd > 0)
	cv = sd[ok] / mu[ok]
	log2_m = np.log2(mu[ok])
	log2_cv = np.log2(cv)

	svr_gamma = 1000. / len(mu[ok])
	clf = SVR(gamma=svr_gamma)
	clf.fit(log2_m[:, np.newaxis], log2_cv)
	fitted_fun = clf.predict
	# Score is the relative position with respect of the fitted curve
	score = log2_cv - fitted_fun(log2_m[:, np.newaxis])
	score = score * valid[ok]
	return np.where(ok)[0][np.argsort(score)][-n_genes:]
