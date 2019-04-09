import loompy
import numpy as np
from typing import *
from .human import cc_genes_human, TFs_human, s_human, g1_human, g2m_human
from .mouse import cc_genes_mouse, TFs_mouse, s_mouse, g1_mouse, g2m_mouse
from types import SimpleNamespace


class Species:
	@staticmethod
	def detect(ds: loompy.LoomConnection) -> Any:  # Really returns Species, but mypy doesn't understand that
		if "species" in ds.attrs:
			name = ds.attrs.species
		elif "Gene" in ds.ra:
			for gene, species in {
				"NOTCH2NL": "Homo sapiens",
				"Tspy1": "Rattus norvegicus",
				"Actb": "Mus musculus",  # Note must come after rat, because rat has the same gene name
				"actb1": "Danio rerio",
				"Act5C": "Drosophila melanogaster",
				"ACT1": "Saccharomyces cerevisiae",
				"act1": "Schizosaccharomyces pombe",
				"act-1": "Caenorhabditis elegans",
				"ACT12": "Arabidopsis thaliana",
				"AFTTAS": "Gallus gallus"
			}.items():
				if gene in ds.ra.Gene:
					name = species
		else:
			raise ValueError("Failed to auto-detect species")

		return Species(name)
		
	def __init__(self, name: str) -> None:
		self.name = name
		if name == "Homo sapiens":
			genes = {
				"TFs": TFs_human,
				"cellcycle": cc_genes_human,
				"sex": ["XIST", "TSIX"],
				"ieg": ['JUNB', 'FOS', 'EGR1', 'JUN'],
				"g1": g1_human,
				"s": s_human,
				"g2m": g2m_human
			}
		elif name == "Mus musculus":
			genes = {
				"TFs": TFs_mouse,
				"cellcycle": cc_genes_mouse,
				"sex": ["Xist", "Tsix"],
				"ieg": ['Junb', 'Fos', 'Egr1', 'Jun'],
				"g1": g1_mouse,
				"s": s_mouse,
				"g2m": g2m_mouse
			}
		else:
			genes = {
				"TFs": [],
				"cellcycle": [],
				"sex": [],
				"ieg": [],
				"g1": [],
				"s": [],
				"g2m": []
			}
		self.genes = SimpleNamespace(**genes)
	
	@staticmethod
	def mask(ds: loompy.LoomConnection, categories: List[str]) -> np.ndarray:
		"""
		Create a boolean mask that includes all genes except those that belong to any of the categories

		Args:
			categories		Any combination of "TFs", "cellcycle", "sex", "ieg", "g1", "s", "g2m"
		"""
		s = Species.detect(ds)
		mask = np.zeros(ds.shape[0], dtype=bool)
		for cat in categories:
			mask = mask | np.isin(ds.ra.Gene, s.genes[cat])
		return mask
