from typing import Union, Callable

import numpy as np
from openTSNE import TSNEEmbedding, affinity, initialization, callbacks


def art_of_tsne(X: np.ndarray, metric: Union[str, Callable] = "euclidean") -> TSNEEmbedding:
	"""
	Implementation of Dmitry Kobak and Philipp Berens "The art of using t-SNE for single-cell transcriptomics" based on openTSNE.
	See https://doi.org/10.1038/s41467-019-13056-x | www.nature.com/naturecommunications
	"""
	n = X.shape[0]
	if n > 100_000:
		# Subsample, optimize, then add the remaining cells and optimize again
		# Also, use exaggeration == 4
		
		# Subsample and run a regular art_of_tsne on the subset
		indices = np.random.permutation(n)
		reverse = np.argsort(indices)
		X_sample, X_rest = X[indices[:n // 40]], X[indices[n // 40:]]
		Z_sample = art_of_tsne(X_sample)

		if isinstance(Z_sample.affinities, affinity.Multiscale):
			rest_init = Z_sample.prepare_partial(X_rest, k=1, perplexities=[1 / 3, 1 / 3])
		else:
			rest_init = Z_sample.prepare_partial(X_rest, k=1, perplexity=1 / 3)
		init_full = np.vstack((Z_sample, rest_init))[reverse]
		init_full = init_full / (np.std(init_full[:, 0]) * 10000)

		# Use multiscale perplexity
		affinities_multiscale_mixture = affinity.Multiscale(
			X,
			perplexities=[30, n / 100],
			metric=metric,
			method="approx",
			n_jobs=-1
		)

		Z = TSNEEmbedding(
			init_full,
			affinities_multiscale_mixture,
			negative_gradient_method="fft",
			n_jobs=-1,
			callbacks=[callbacks.ErrorLogger()]
		)
		Z.optimize(n_iter=250, inplace=True, exaggeration=12, momentum=0.5, learning_rate=n / 12, n_jobs=-1)
		Z.optimize(n_iter=750, inplace=True, exaggeration=4, momentum=0.8, learning_rate=n / 12, n_jobs=-1)
	elif n > 3_000:
		# Use multiscale perplexity
		affinities_multiscale_mixture = affinity.Multiscale(
			X,
			perplexities=[30, n / 100],
			metric=metric,
			method="approx",
			n_jobs=8
		)
		init = initialization.pca(X)
		Z = TSNEEmbedding(
			init,
			affinities_multiscale_mixture,
			negative_gradient_method="fft",
			n_jobs=-1,
			callbacks=[callbacks.ErrorLogger()]
		)
		Z.optimize(n_iter=250, inplace=True, exaggeration=12, momentum=0.5, learning_rate=n / 12, n_jobs=-1)
		Z.optimize(n_iter=750, inplace=True, exaggeration=1, momentum=0.8, learning_rate=n / 12, n_jobs=-1)
	else:
		# Just a plain TSNE with high learning rate
		lr = max(200, n / 12)
		aff = affinity.PerplexityBasedNN(
			X,
			perplexity=30,
			metric=metric,
			method="approx",
			n_jobs=-1
		)

		init = initialization.pca(X)

		Z = TSNEEmbedding(
			init,
			aff,
			learning_rate=lr,
			n_jobs=-1,
			negative_gradient_method="fft",
			callbacks=[callbacks.ErrorLogger()]
		)
		Z.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
		Z.optimize(750, exaggeration=1, momentum=0.5, inplace=True)
	return Z
