
import numpy as np
from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from pynndescent import NNDescent


def art_of_tsne(X: np.ndarray) -> TSNEEmbedding:
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
		Xsub = X[indices[:n / 40], :]
		Zsub = art_of_tsne(Xsub)

		# Find single nearest neighbor using pynndescent
		nn, _ = NNDescent(data=Xsub, metric="euclidean", n_jobs=-1).query(X, k=1, queue_size=min(5.0, n))
		nn = nn[:, 1:]
		init = Zsub[nn]  # initialize all points to the nearest point in the subsample

		# Use multiscale perplexity
		affinities_multiscale_mixture = affinity.Multiscale(
			X,
			perplexities=[30, n / 100],
			metric="euclidean",
			n_jobs=8
		)
		Z = TSNEEmbedding(
			init,
			affinities_multiscale_mixture,
			negative_gradient_method="fft",
			n_jobs=-1,
		)
		Z.optimize(n_iter=250, inplace=True, exaggeration=12, momentum=0.5, learning_rate=n / 12, n_jobs=-1)
		Z.optimize(n_iter=750, inplace=True, exaggeration=4, momentum=0.8, learning_rate=n / 12, n_jobs=-1)
	elif n > 3_000:
		# Use multiscale perplexity
		affinities_multiscale_mixture = affinity.Multiscale(
			X,
			perplexities=[30, n / 100],
			metric="euclidean",
			n_jobs=8
		)
		init = initialization.pca(X)
		Z = TSNEEmbedding(
			init,
			affinities_multiscale_mixture,
			negative_gradient_method="fft",
			n_jobs=-1,
		)
		Z.optimize(n_iter=250, inplace=True, exaggeration=12, momentum=0.5, learning_rate=n / 12, n_jobs=-1)
		Z.optimize(n_iter=750, inplace=True, exaggeration=1, momentum=0.8, learning_rate=n / 12, n_jobs=-1)
	else:
		# Just a plain TSNE with high learning rate
		Z = TSNE(perplexity=30, metric="euclidean", n_jobs=-1, initialization="pca", learning_rate=n / 12).fit(X)
	return Z
