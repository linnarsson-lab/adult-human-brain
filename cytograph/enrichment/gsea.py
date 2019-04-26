import logging
from typing import List

import numpy as np

import loompy


class GSEA:
	"""
	Gene set enrichment analysis according to http://software.broadinstitute.org/gsea/doc/subramanian_tamayo_gsea_pnas.pdf
	
	Note: we rank genes based on the enrichment score from MarkerSelection, not correlation as in the paper.
	Note: currently, FDR correction not implemented; the fit() method returns uncorrected P values
	"""
	def __init__(self) -> None:
		pass

	def fit(self, dsagg: loompy.LoomConnection, gene_set_file: str) -> np.ndarray:
		"""
		Test each gene set (columns in gene_sets) for enrichment in each column of dsagg

		Args:
			dsagg:			An aggregated loom file with a layer "enrichment", shape (N, M)
			gene_set_file:	Path to a gene set file in gmt format (http://software.broadinstitute.org/gsea/downloads.jsp)
		
		Returns:
			p_values:		An ndarray of shape (K, M), where K is number of gene sets, indicating the P value for each test
			gene_set_names:	A list of K gene set names
		"""
		# Load the gene sets
		N = dsagg.shape[0]
		gene_set_names: List[str] = []
		K = 0  # The number of gene sets
		with open(gene_set_file) as f:
			for line in f.readlines():
				items = line[:-1].split("\t")
				gene_set_names.append(items[0])
				K += 1
		gene_sets = np.zeros((N, K), dtype='bool')
		k = 0
		with open(gene_set_file) as f:
			for line in f.readlines():
				items = line[:-1].split("\t")
				accessions = [f"ENSMUSG{int(gid):011d}" for gid in items[2:]]
				gene_sets[np.in1d(dsagg.ra.Accession, accessions).nonzero()[0], k] = True
				k += 1

		# Compute p values by permutation test
		M = dsagg.shape[1]
		p_values = np.zeros((K, M))
		for k in range(K):
			logging.info(f"GSEA of {M} clusters for {gene_set_names[k]} ({k}/{K})")
			for m in range(M):
				data = dsagg["enrichment"][:, m]
				null_dist = np.zeros(1000)
				for ix in range(1000):
					null_dist[ix] = self._fit_one(data, np.random.permutation(gene_sets[:, k]))
				null_dist.sort()
				score = self._fit_one(data, gene_sets[:, k])
				p_values[k, m] = 1 - np.searchsorted(null_dist, score) / 1000
		return (p_values, gene_set_names)

	def _fit_one(self, enrichment: np.ndarray, gene_set: np.ndarray) -> float:
		"""
		Test an array of (cytograph) enrichment scores against a single gene set

		See Appendix of http://software.broadinstitute.org/gsea/doc/subramanian_tamayo_gsea_pnas.pdf
		"""
		p_hit = np.cumsum(enrichment * gene_set)
		p_hit = p_hit / p_hit[-1]
		p_miss = np.cumsum(~gene_set)
		p_miss = p_miss / p_miss[-1]
		score = (p_hit - p_miss).max()
		return score
