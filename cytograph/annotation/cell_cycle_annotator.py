from typing import Tuple

import numpy as np

import loompy
from cytograph.species import Species


class CellCycleAnnotator:
	def __init__(self, species: Species) -> None:
		self.species = species

	def fit(self, ds: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		g1_indices = np.isin(ds.ra.Gene, self.species.genes.g1)
		s_indices = np.isin(ds.ra.Gene, self.species.genes.s)
		g2m_indices = np.isin(ds.ra.Gene, self.species.genes.g2m)
		g1_totals = ds[g1_indices, :].sum(axis=0)
		s_totals = ds[s_indices, :].sum(axis=0)
		g2m_totals = ds[g2m_indices, :].sum(axis=0)
		if "TotalUMIs" in ds.ca:
			total_umis = ds.ca.TotalUMIs  # From loompy
		else:
			total_umis = ds.ca.TotalUMI  # From cytograph
		return (g1_totals / total_umis, s_totals / total_umis, g2m_totals / total_umis)
	
	def annotate(self, ds: loompy.LoomConnection) -> None:
		"""
		Compute the fraction of UMIs that arise from cell cycle genes
		"""
		(g1, s, g2m) = self.fit(ds)
		ds.ca.CellCycle_G1 = g1
		ds.ca.CellCycle_S = s
		ds.ca.CellCycle_G2M = g2m
		ds.ca.CellCycle = (g1 + s + g2m)
		ds.ca.IsCycling = ds.ca.CellCycle > 0.01  # This threshold will depend on the species and on the gene list; 1% is good for the current human and mouse gene sets
