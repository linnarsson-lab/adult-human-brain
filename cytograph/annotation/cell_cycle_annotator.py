import loompy
import numpy as np
from typing import *
import matplotlib.pyplot as plt
from cytograph.species import Species


class CellCycleAnnotator:
	def __init__(self, species: Species, layer: str = None) -> None:
		self.layer = layer
		self.species = species

	def _totals_per_cell(self, ds: loompy.LoomConnection, layer: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		g1_indices = np.isin(ds.ra.Gene, self.species.genes.g1)
		s_indices = np.isin(ds.ra.Gene, self.species.genes.s)
		g2m_indices = np.isin(ds.ra.Gene, self.species.genes.g2m)
		g1_totals = ds[layer][g1_indices, :].sum(axis=0)
		s_totals = ds[layer][s_indices, :].sum(axis=0)
		g2m_totals = ds[layer][g2m_indices, :].sum(axis=0)
		return (g1_totals, s_totals, g2m_totals)
	
	def fit(self, ds: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Compute total expression of genes specific for G1, S and G2/M phases of the cell cycle

		Returns:
			g1		Sum of expression of G1 phase genes
			s		Sum of expression of S phase genes
			g2m		Sum of expression of G2/M phase genes
		"""
		if self.layer is None:
			if "spliced_exp" in ds.layers:
				layer = "spliced_exp"
			elif "pooled" in ds.layers:
				layer = "pooled"
			else:
				layer = ""

		g1_indices = np.isin(ds.ra.Gene, self.species.genes.g1)
		s_indices = np.isin(ds.ra.Gene, self.species.genes.s)
		g2m_indices = np.isin(ds.ra.Gene, self.species.genes.g2m)
		g1 = ds[layer][g1_indices, :].sum(axis=0)
		s = ds[layer][s_indices, :].sum(axis=0)
		g2m = ds[layer][g2m_indices, :].sum(axis=0)

		return (g1, s, g2m)
	
	def annotate(self, ds: loompy.LoomConnection) -> None:
		(g1, s, g2m) = self.fit(ds)
		ds.ca.CellCycle_G1 = g1
		ds.ca.CellCycle_S = s
		ds.ca.CellCycle_G2M = g2m
