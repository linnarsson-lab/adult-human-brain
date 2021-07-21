from typing import List
import numpy as np
import loompy
import logging
from cytograph.preprocessing import div0


class FeatureSelectionByDeviance():
    def __init__(self, n_genes: int = 2000, mask: List[str] = None) -> None:
        """
        Args:
            n_genes		Number of genes to select
            mask		Optional list indicating categories of genes that should not be selected
        """
        self.n_genes = n_genes
        self.mask = mask

    def fit(self, ds: loompy.LoomConnection) -> np.ndarray:
        """
        Selects genes that show high deviance using a binomial model
        Args:
            ws:	shoji.Workspace containing the data to be used
        Returns:
            ndarray of indices of selected genes

        Remarks:
            If the tensor "ValidGenes" exists, only ValidGenes == True genes will be selected
            See equation D_j on p. 14 of https://doi.org/10.1186/s13059-019-1861-6
        """
        logging.info(" DevianceStatistics: Computing the variance of Pearson residuals (deviance)")
        totals = ds.ca.TotalUMI
        gene_totals = np.sum(ds[:, :], axis=1)
        overall_totals = ds.ca.TotalUMI.sum()
        
        batch_size = 1000
        ix = 0
        n_cells = ds.shape[1]
        acc = np.zeros(ds[:, :].T.shape)
        while ix < n_cells:
            data = ds[:, ix: ix + batch_size].T
            expected = totals[ix: ix + batch_size, None] @ div0(gene_totals[None, :], overall_totals)
            residuals = div0((data - expected), np.sqrt(expected + np.power(expected, 2) / 100))
            acc[ix: ix + batch_size, :] = residuals
            ix += batch_size
        # acc = np.clip(acc, -np.sqrt(n_cells), np.sqrt(n_cells))
        d = np.var(acc, axis=0)
        ds.ra.ResidualVariance = d

        logging.info(" FeatureSelectionByDeviance: Removing invalid and masked genes")
        valid = ds.ra.Valid == 1
        if self.mask is not None:
            valid = np.logical_and(valid, np.logical_not(self.mask))

        temp = []
        for gene in np.argsort(-d):
            if valid[gene]:
                temp.append(gene)
            if len(temp) >= self.n_genes:
                break
        genes = np.array(temp)
        logging.info(f" FeatureSelectionByDeviance: Selected the top {len(genes)} genes")
        selected = np.zeros(ds.shape[0], dtype=bool)
        selected[np.sort(genes)] = True
        return selected

    def select(self, ds: loompy.LoomConnection) -> np.ndarray:
        selected = self.fit(ds)
        ds.ra.Selected = selected.astype('int')
        return selected