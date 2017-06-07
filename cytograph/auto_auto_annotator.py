from typing import *
import os
import logging
import numpy as np
import pandas as pd
import re
import loompy


class AutoAutoAnnotator:
	def __init__(self, pep: float = 0.05) -> None:
		self.pep = pep
	
	def fit(self, dsagg: loompy.LoomConnection) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
		data = dsagg.layer["trinaries"][:, :]
		n_clusters = data.shape[1]
		genes = np.where(np.logical_and(data.sum(axis=1) < n_clusters * 0.5, data.sum(axis=1) > 0))[0]
		data = data[genes, :]

		# shape (n_clusters, n_genes, n_genes)
		scores = np.zeros((data.shape[1], data.shape[0], data.shape[0]))
		for ix in range(data.shape[1]):
			trinaries = data[:, ix]
			scores[ix, :, :] = np.tril(np.outer(trinaries, trinaries), k=-1)

		positives = np.where(scores > self.pep, 1, 0)
		candidates = np.where(np.logical_and(positives.sum(axis=0) > 0, positives.sum(axis=0) < 5))
		specificity = 1 / positives.sum(axis=0)[candidates]

		result = []
		for ix in range(data.shape[1]):
			mask = positives[ix, candidates[0], candidates[1]] == 1
			temp = (candidates[0][mask], candidates[1][mask], specificity[mask])
			ordering = np.argsort(-temp[2])
			temp = (genes[temp[0][ordering]], genes[temp[1][ordering]], temp[2][ordering])
			result.append(temp)
		return result
