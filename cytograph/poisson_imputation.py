import numpy as np
import scipy.sparse as sparse
from sklearn.neighbors import NearestNeighbors
import logging
import loompy
import cytograph as cg


class PoissonImputation:
    def __init__(self, k: int, N: int, n_genes: int, n_components: int) -> None:
        self.k = k
        self.N = N
        self.n_genes = n_genes
        self.n_components = n_components

    def impute_inplace(self, ds: loompy.LoomConnection) -> np.ndarray:
        # Select genes
        genes = cg.FeatureSelection(n_genes=self.n_genes).fit(ds)
        
        # Factorize
        logging.info("Factorizing by HPF")
        data = ds.sparse(genes=genes).T
        hpf = cg.HPF(k=self.n_components)
        hpf.fit(data)
        
        # Compute KNN matrix
        logging.info("Computing nearest neighbors")
        nn = NearestNeighbors(self.k, algorithm="auto", metric='correlation', n_jobs=4)
        nn.fit(hpf.theta)
        knn = nn.kneighbors_graph(hpf.theta, mode='connectivity')  # Returns a CSR sparse graph, including self-edges
        
        # Compute size-corrected Poisson MLE rates
        size_factors = ds.map([np.sum], axis=1)[0]

        logging.info("Imputing values in place")
        ix = 0
        window = 400
        while ix < ds.shape[0]:
            # Load the data for a subset of genes
            data = ds[ix:min(ds.shape[0], ix + window), :]
            data_std = data / size_factors
            # Sum of MLE rates for neighbors
            imputed = knn.dot(data_std.T).T
            ds[ix:min(ds.shape[0], ix + window), :] = imputed.astype('float32')
            ix += window
