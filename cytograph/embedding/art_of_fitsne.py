import logging
from typing import Callable, Union

import numpy as np
from fitsne import FItSNE as fitsne
from annoy import AnnoyIndex


def art_of_tsne(X: np.ndarray, metric: Union[str, Callable] = "euclidean") -> np.ndarray:
	"""
	Implementation of Dmitry Kobak and Philipp Berens "The art of using t-SNE for single-cell transcriptomics" based on FItSNE.
	See https://doi.org/10.1038/s41467-019-13056-x | www.nature.com/naturecommunications
	"""
	n, d = X.shape
	if n > 100_000:
		# Downsample, optimize, then add the remaining cells and optimize again
		logging.info(f"Creating subset of {n // 40} elements")
		ns = n // 40
		pca_init = X[:, :2] / np.std(X[:, 0]) * 0.0001
		X_sample = np.random.choice(n, ns, replace=False)
		logging.info(f"Embedding the subset")
		Z_sample = fitsne(X[X_sample, :], perplexity_list=[30, ns // 100], initialization = pca_init[X_sample, :], seed=42, learning_rate = ns / 12)

		logging.info(f"Building an annoy index of the subset")
		annoy = AnnoyIndex(d, 'euclidean')
		for i in range(ns):
			annoy.add_item(i, X_sample[i, :])
		annoy.build(10)
		logging.info(f"Mapping the full set of {n} elements to nearest subset elements")
		X_init = np.zeros((n, 2))
		for i in range(n):
			X_init[i, :] = np.median(Z_sample[annoy.get_nns_by_vector(X[i, :], 10, search_k=-1, include_distances=False), :])
		X_init = X_init / np.std(X_init[:, 0]) * 0.0001

		logging.info(f"Creating the full tSNE embedding")
		return fitsne(X, perplexity=30, initialization=X_init, late_exag_coeff=4, start_late_exag_iter=250, learning_rate=n / 12, seed=42)
	elif n > 3_000:
		pca_init = X[:, :2] / np.std(X[:, 0]) * 0.0001
		return fitsne(X, perplexity_list=[30, n // 100], initialization = pca_init, seed=42, learning_rate = ns / 12)
	else:
		pca_init = X[:, :2] / np.std(X[:, 0]) * 0.0001
		return fitsne(X, perplexity=30, initialization = pca_init, seed=42, learning_rate = ns / 12)


import logging
import sys
logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=20)
logging.captureWarnings(True)
art_of_tsne(np.random.uniform(size=(101000, 5)))
