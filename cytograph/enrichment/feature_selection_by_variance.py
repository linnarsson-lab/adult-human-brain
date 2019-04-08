import logging
import numpy as np
from typing import *
from sklearn.svm import SVR
import loompy


class FeatureSelectionByVariance:
	def __init__(self, n_genes: int, layer: str = "", mask: np.ndarray = None) -> None:
		self.n_genes = n_genes
		self.layer = layer
		self.mask = mask

	def fit(self, ds: loompy.LoomConnection) -> np.ndarray:
		"""
		Fits a noise model (CV vs mean)

		Args:
			ds (LoomConnection):	Dataset

		Returns:
			ndarray of selected genes (bool array)
		
		Remarks:
			If the row attribute "Valid" exists, only Valid == 1 genes will be selected
		"""
		(mu, sd) = ds[self.layer].map((np.mean, np.std), axis=0)

		if "Valid" in ds.ra:
			valid = ds.ra.Valid == 1
		else:
			valid = np.ones(ds.shape[0], dtype='bool')
		if self.mask is not None:
			valid = np.logical_and(valid, np.logical_not(self.mask))
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
		genes = np.where(ok)[0][np.argsort(score)][-self.n_genes:]
		selected = np.zeros(ds.shape[0], dtype=bool)
		selected[np.sort(genes)] = True
		return selected
	
	def select(self, ds: loompy.LoomConnection) -> np.ndarray:
		selected = self.fit(ds)
		ds.ra.Selected = selected.astype("int")
		return selected
