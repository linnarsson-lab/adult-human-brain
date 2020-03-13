import loompy
import numpy as np
import logging
from cytograph.preprocessing import Normalizer


class BinaryDifferentialExpression:
    def __init__(self, sig_thresh: float = 0.001, fnnz: np.ndarray = None, mu: np.ndarray = None) -> None:
        self.sig_thresh = sig_thresh
        self.fnnz = fnnz
        self.mu = mu
        self.labels: np.ndarray = None
        self.n_labels: int = None
        self.stats: None

    def fit(self, ds: loompy.LoomConnection, labels: np.ndarray = None):

        from diffxpy.api.test import pairwise

        logging.info('Calculating differential expression statistics')

        self.labels = labels
        self.n_labels = len(np.unique(labels))

        normalizer = Normalizer(False)
        X = normalizer.fit_transform(ds, ds[:, :])

        self.stats = pairwise(
            data=X.T,
            grouping=self.labels,
            test='t-test',
            lazy=False,
            gene_names=ds.ra.Gene,
            noise_model=None,
            sample_description=ds.ca.CellID,
            pval_correction='by_test',
            is_logged=True
        )

        if self.fnnz is None:
            logging.debug('Calculating fnnz for each label')
            # Number of cells per cluster
            _, renumbered = np.unique(self.labels, return_inverse=True)
            sizes = np.bincount(renumbered, minlength=self.n_labels)
            # Number of nonzero values per cluster
            nnz = ds.aggregate(None, None, self.labels, np.count_nonzero, None)
            # Scale by number of cells
            self.fnnz = nnz / sizes
        if self.mu is None:
            logging.debug('Calculating means for each label')
            # Mean value per cluster
            self.mu = ds.aggregate(None, None, self.labels, "mean", None)
            
    def select(self, c1, c2):

        logging.debug(f'Finding binary DE genes for labels {c1} and {c2}')

        # test for differential expression
        q_val = self.stats.qval_pairs(groups0=[c1], groups1=[c2])[0][0] < self.sig_thresh
        # test fold change >= 2
        fold_change = np.logical_or(self.mu[:, c1] / self.mu[:, c2] >= 2, self.mu[:, c2] / self.mu[:, c1] >= 2)
        # test for > 30% in upregulated cluster; difference > 70% of max
        fraction = np.zeros(fold_change.shape).astype('bool')
        m = np.max(np.vstack((self.fnnz[:, c1], self.fnnz[:, c2])), axis=0)
        ix = self.mu[:, c1] > self.mu[:, c2]
        fraction[ix] = (self.fnnz[:, c1][ix] > 0.3) & (( (self.fnnz[:, c1][ix] - self.fnnz[:, c2][ix]) / m[ix] ) > 0.7)
        ix = self.mu[:, c1] < self.mu[:, c2]
        fraction[ix] = (self.fnnz[:, c2][ix] > 0.3) & (( (self.fnnz[:, c2][ix] - self.fnnz[:, c1][ix]) / m[ix] ) > 0.7)

        # Add tests and return
        selected = q_val & fold_change & fraction
        return selected

    def count(self):

        logging.info(f'Calculating DE matrix for {self.n_labels} labels')

        de_matrix = np.zeros((self.n_labels, self.n_labels))

        for c1 in range(self.n_labels):
            for c2 in range(self.n_labels):
                if c2 <= c1:
                    continue
                de_matrix[c1, c2] = self.select(c1, c2).sum()
    
        return de_matrix + de_matrix.T
