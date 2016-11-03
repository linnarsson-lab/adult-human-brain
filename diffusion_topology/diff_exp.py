
import numpy as np
from scipy.stats import mannwhitneyu


def is_gene_enriched(ds, cells1, cells2, gene_name):
	norm = ds.col_attrs["_TotalRNA"]
	a = np.log(ds[np.where(ds.Gene == gene), :][0, cells1]+1)/norm
	b = np.log(ds[np.where(ds.Gene == gene), :][0, cells2]+1)/norm

	(_, pval) = mannwhitneyu(a, b, alternative="greater")
	return pval
