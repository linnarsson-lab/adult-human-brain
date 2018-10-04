import cytograph as cg
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold.t_sne import _joint_probabilities_nn
from typing import *
import numpy as np
from pynndescent import NNDescent
from .absolute.metrics import jensen_shannon_distance


def tsne_js(X: np.ndarray, *, n_components: int = 2, dof: int = 1, perplexity: int = 30, distances_nn: np.ndarray = None, neighbors_nn: np.ndarray = None, radius: float = 0.2) -> np.ndarray:
	"""
	Exploit the internals of sklearn TSNE to efficiently compute TSNE with Jensen-Shannon
	distance metric. This will make use of multiple cores for the nearest-neighbor search.

	Args:
		X					Input matrix (n_samples, n_features)
		n_components		Number of components in embedding (2 or 3)
		dof					Degrees of freedom (n_components - 1 in standard TSNE)
		perplexity			Perplexity
		distances_nn		Precomputed nearest-neighbor distances (or None)
		neighbors_nn		Precomputed nearest-neighbor indices (or None)
		radius				Max distance
	
	Remarks:
		If distances_nn and neighbors_nn are given, they will be used as-is to calculate the t-SNE P matrix
		(this will also implictly set k)
	"""
	n_samples = X.shape[0]

	if distances_nn is None or neighbors_nn is None:
		k = min(n_samples - 1, int(3. * perplexity + 1))
		nn = NNDescent(data=X, metric=jensen_shannon_distance)
		indices_nn, distances_nn = nn.query(X, k=k)
		indices_nn = indices_nn[:, 1:]
		distances_nn = distances_nn[:, 1:]

	distances_nn[distances_nn > radius] = 1
	P = _joint_probabilities_nn(distances_nn, indices_nn, perplexity, False)
	
	pca = PCA(n_components=n_components, svd_solver='randomized')
	X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)

	tsne = TSNE(n_components=n_components, perplexity=perplexity)
	degrees_of_freedom = max(n_components - 1, 1)
	return tsne._tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded, neighbors=indices_nn, skip_num_points=0)
