import loompy
import numpy as np
from typing import *
import matplotlib.pyplot as plt
import cytograph as cg


class CellCycleAnnotator:
	def __init__(self, ds: loompy.LoomConnection) -> None:
		self.ds = ds
		species = cg.species(ds)
		if species == "Homo sapiens":
			# Cell cycle gene sets from Tirosh et al. doi:10.1126/science.aad0501, plus histones for S phase from own analyses
			# Removed BIRC5 because it's expressed in certain non-cycling heart cells
			self.g1 = ["MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8"]
			self.s = ["HIST1H2AJ", "HIST1H2AM", "HIST1H1D", "HIST1H4C"]
			self.g2m = ["HMGB2", "CDK1", "NUSAP1", "UBE2C", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "HJURP", "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"]
		elif species == "Mus musculus":
			self.g1 = ['Mcm5', 'Pcna', 'Tyms', 'Fen1', 'Mcm2', 'Mcm4', 'Rrm1', 'Ung', 'Gins2', 'Mcm6', 'Cdca7', 'Dtl', 'Prim1', 'Uhrf1', 'Cenpu', 'Hells', 'Rfc2', 'Rpa2', 'Nasp', 'Rad51ap1', 'Gmnn', 'Wdr76', 'Slbp', 'Ccne2', 'Ubr7', 'Pold3', 'Msh2', 'Atad2', 'Rad51', 'Rrm2', 'Cdc45', 'Cdc6', 'Exo1', 'Tipin', 'Dscc1', 'Blm', 'Casp8ap2', 'Usp1', 'Clspn', 'Pola1', 'Chaf1b', 'Brip1', 'E2f8']
			self.s = ["Hist1h2aj", "Hist1h2am", "Hist1h1d", "Hist1h4c"]
			self.g2m = ['Hmgb2', 'Cdk1', 'Nusap1', 'Ube2c', 'Tpx2', 'Top2a', 'Ndc80', 'Cks2', 'Nuf2', 'Cks1b', 'Mki67', 'Tmpo', 'Cenpf', 'Tacc3', 'Fam64a', 'Smc4', 'Ccnb2', 'Ckap2l', 'Ckap2', 'Aurkb', 'Bub1', 'Kif11', 'Anp32e', 'Tubb4b', 'Gtse1', 'Kif20b', 'Hjurp', 'Hjurp', 'Cdca3', 'Hn1', 'Cdc20', 'Ttk', 'Cdc25c', 'Kif2c', 'Rangap1', 'Ncapd2', 'Dlgap5', 'Cdca2', 'Cdca8', 'Ect2', 'Kif23', 'Hmmr', 'Aurka', 'Psrc1', 'Anln', 'Lbr', 'Ckap5', 'Cenpe', 'Ctcf', 'Nek2', 'G2e3', 'Gas2l3', 'Cbx5', 'Cenpa']

	def totals_per_cell(self, layer: str = "") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		g1_indices = np.isin(self.ds.ra.Gene, self.g1)
		s_indices = np.isin(self.ds.ra.Gene, self.s)
		g2m_indices = np.isin(self.ds.ra.Gene, self.g2m)
		g1_totals = self.ds[layer][g1_indices, :].sum(axis=0)
		s_totals = self.ds[layer][s_indices, :].sum(axis=0)
		g2m_totals = self.ds[layer][g2m_indices, :].sum(axis=0)
		return (g1_totals, s_totals, g2m_totals)
	
	def annotate_loom(self) -> None:
		(g1, s, g2m) = self.totals_per_cell("spliced_exp")
		cycling = (g1 + s + g2m) > 1
		self.ds.ca.Cycling = cycling.astype("int")
		self.ds.ca.G1 = g1
		self.ds.ca.S = s
		self.ds.ca.G2M = g2m

	def plot_cell_cycle(self, path: str) -> None:
		(g1, s, g2m) = self.totals_per_cell("")
		ordering = np.random.permutation(self.ds.shape[1])
		tsne_x = self.ds.ca.TSNE[:, 0][ordering]
		tsne_y = self.ds.ca.TSNE[:, 1][ordering]
		g1 = g1[ordering]
		s = s[ordering]
		g2m = g2m[ordering]
		colors = cg.colorize(self.ds.ca.Clusters)[ordering]

		plt.figure(figsize=(20, 4))
		plt.subplot(141)
		cells = (g1 + s + g2m) > np.percentile((g1 + s + g2m), 99) / 5
		plt.scatter(tsne_x, tsne_y, c='lightgrey', marker='.', lw=0)
		plt.scatter(tsne_x[cells], tsne_y[cells], c=colors[cells], marker='.', lw=0)
		plt.title("Clusters")
		plt.subplot(142)
		cells = g1 > np.percentile(g1, 99) / 5
		plt.scatter(tsne_x, tsne_y, c='lightgrey', marker='.', lw=0)
		plt.scatter(tsne_x[cells], tsne_y[cells], c=g1[cells], marker='.', lw=0, s=30, alpha=0.7)
		plt.title("G1")
		plt.subplot(143)
		cells = s > np.percentile(s, 99) / 5
		plt.scatter(tsne_x, tsne_y, c='lightgrey', marker='.', lw=0)
		plt.scatter(x=tsne_x[cells], y=tsne_y[cells], c=s[cells], marker='.', lw=0, s=30, alpha=0.7)
		plt.title("S")
		plt.subplot(144)
		cells = g2m > np.percentile(g2m, 99) / 5
		plt.scatter(tsne_x, tsne_y, c='lightgrey', marker='.', lw=0)
		plt.scatter(tsne_x[cells], tsne_y[cells], c=g2m[cells], marker='.', lw=0, s=30, alpha=0.7)
		plt.title("G2/M")
		plt.savefig(path, dpi=144)

