import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, laplacian


class DDRTree:
    def __init__(self, lmbda: float, sigma: float, gamma: float) -> None:
        self.lmbda = lmbda
        self.sigma = sigma
        self.gamma = gamma
    
    def fit(self, X: np.ndarray) -> None:
        # This code closely follows Algorithm 2 of http://dx.doi.org/10.1145/2783258.2783309
        Z = PCA(n_components=2).fit_transform(X)  # (n_samples, 2)
        N = X.shape[0]
        K = N if N < 100 else 200 * np.log(N) / (np.log(N) + np.log(100))
        Y = Z.copy()  # (n_samples, 2)
        d = squareform(pdist(Y))  # (n_samples, n_samples)
        mst = minimum_spanning_tree(d)
        L = laplacian(mst)
        # This oneliner is explained here: https://medium.com/dataholiks-distillery/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        xy_dists = -2 * np.dot(Z, Y.T) + np.sum(Y**2,    axis=1) + np.sum(Z**2, axis=1)[:, np.newaxis]
        R = xy_dists / xy_dists.sum(axis=1)[:, None]  # (n_samples, n_samples)
        gamma = np.diag(R.sum(axis=1))
        