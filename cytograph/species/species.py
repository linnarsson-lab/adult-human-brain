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
				"g2m": g2m_human,
				"mt": ['MT-CYB', 'MT-ND6', 'MT-CO3', 'MT-ND1', 'MT-ND4', 'MT-CO1', 'MT-ND2', 'MT-CO2', 'MT-ATP8', 'MT-ND4L', 'MT-ATP6', 'MT-ND5', 'MT-ND3']
			}
		elif name == "Mus musculus":
			genes = {
				"TFs": TFs_mouse,
				"cellcycle": cc_genes_mouse,
				"sex": ["Xist", "Tsix"],
				"ieg": ['Junb', 'Fos', 'Egr1', 'Jun'],
				"g1": g1_mouse,
				"s": s_mouse,
				"g2m": g2m_mouse,
				"mt": ['mt-Nd1', 'mt-Nd2', 'mt-Co1', 'mt-Co2', 'mt-Atp8', 'mt-Atp6', 'mt-Co3', 'mt-Nd3', 'mt-Nd4l', 'mt-Nd4', 'mt-Nd5', 'mt-Cytb', 'mt-Nd6']
			}
		else:
			genes = {
				"TFs": [],
				"cellcycle": [],
				"sex": [],
				"ieg": [],
				"g1": [],
				"s": [],
				"g2m": [],
				"mt": []
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
			mask = mask | np.isin(ds.ra.Gene, s.genes.__dict__[cat])
		return mask
