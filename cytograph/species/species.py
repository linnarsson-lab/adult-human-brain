import loompy
from .human import cc_genes_human, TFs_human
from .mouse import cc_genes_mouse, TFs_mouse


class Species:
	def __init__(self, ds: loompy.LoomConnection) -> None:
		self.name = "Unknown"
		if "Gene" in ds.ra:
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
					self.name = species
		if self.name == "Homo sapiens":
			self.cell_cycle_genes = cc_genes_human
			self.TFs = TFs_human
			self.sex_genes = ["XIST", "TSIX"]
		elif self.name == "Mus musculus":
			self.cell_cycle_genes = cc_genes_mouse
			self.TFs = TFs_mouse
			self.sex_genes = ["Xist", "Tsix"]
		else:
			self.cell_cycle_genes = []
			self.TFs = []
			self.sex_genes = []
