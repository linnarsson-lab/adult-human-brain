import cytograph as cg
import numpy as np
import scipy.sparse as sparse
from scipy.interpolate import griddata
from scipy.stats import poisson
import loompy
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree, NearestNeighbors
import logging
from sklearn.manifold import TSNE
from umap import UMAP
from pynndescent import NNDescent
from sklearn.preprocessing import normalize
from typing import *
import os
import community
import networkx as nx
from .velocity_inference import fit_gamma
from .identify_technical_factors import identify_technical_factors
from .metrics import jensen_shannon_distance
from .cell_cycle_annotator import CellCycleAnnotator
from .velocity_embedding import VelocityEmbedding
from .neighborhood_enrichment import NeighborhoodEnrichment
from ..tsne import tsne
from ..utils import species


cc_genes_human = np.array([
	'ABHD3', 'AC016205.1', 'AC073529.1', 'AC084033.3', 'AC087632.1',
	'AC091057.6', 'AC097534.2', 'AC099850.2', 'AC135586.2', 'ACAA2',
	'ACADM', 'ACP1', 'ACTL6A', 'ACYP1', 'ADCY3', 'ADD3', 'ADK', 'AHCY',
	'AKIRIN2', 'AKR7A2', 'AL359513.1', 'AL449266.1', 'AL513165.2',
	'ANAPC11', 'ANLN', 'ANP32A', 'ANP32B', 'ANP32E', 'AP001347.1',
	'APOLD1', 'ARHGAP11A', 'ARHGEF39', 'ARID1A', 'ARL6IP1', 'ARL6IP6',
	'ARMC1', 'ARPP19', 'ASF1B', 'ASPM', 'ASRGL1', 'ATAD2', 'ATAD5',
	'ATP1B3', 'AURKA', 'AURKB', 'BANF1', 'BARD1', 'BAZ1A', 'BAZ1B',
	'BIRC5', 'BLM', 'BORA', 'BRCA1', 'BRCA2', 'BRD8', 'BRIP1', 'BTG3',
	'BUB1', 'BUB1B', 'BUB3', 'C11orf58', 'C19orf48', 'C1orf112',
	'C1orf35', 'C21orf58', 'C5orf34', 'CACYBP', 'CALM2', 'CAMTA1',
	'CARHSP1', 'CBX1', 'CBX3', 'CBX5', 'CCAR1', 'CCDC14', 'CCDC167',
	'CCDC18', 'CCDC34', 'CCDC77', 'CCNA1', 'CCNA2', 'CCNB1', 'CCNB2',
	'CCNE2', 'CCNF', 'CCT4', 'CCT5', 'CDC20', 'CDC25B', 'CDC25C',
	'CDC27', 'CDC45', 'CDC6', 'CDC7', 'CDCA2', 'CDCA3', 'CDCA4',
	'CDCA7L', 'CDCA8', 'CDK1', 'CDK19', 'CDK2', 'CDK4', 'CDK5RAP2',
	'CDKAL1', 'CDKN1B', 'CDKN2C', 'CDKN3', 'CDT1', 'CENPA', 'CENPC',
	'CENPE', 'CENPF', 'CENPH', 'CENPI', 'CENPJ', 'CENPK', 'CENPL',
	'CENPM', 'CENPN', 'CENPO', 'CENPP', 'CENPQ', 'CENPU', 'CENPW',
	'CENPX', 'CEP112', 'CEP128', 'CEP135', 'CEP192', 'CEP295', 'CEP55',
	'CEP57', 'CEP57L1', 'CEP70', 'CETN3', 'CFAP20', 'CFL2', 'CGGBP1',
	'CHAF1A', 'CHCHD2', 'CHEK1', 'CHEK2', 'CHRAC1', 'CIP2A', 'CIT',
	'CKAP2', 'CKAP2L', 'CKAP5', 'CKLF', 'CKS1B', 'CKS2', 'CLSPN',
	'CMC2', 'CMSS1', 'CNIH4', 'CNN3', 'CNTLN', 'CNTRL', 'COA1',
	'COMMD4', 'COX8A', 'CSE1L', 'CTCF', 'CTDSPL2', 'CWF19L2', 'CYB5B',
	'CYCS', 'DACH1', 'DBF4', 'DBF4B', 'DBI', 'DCAF7', 'DCP2', 'DCXR',
	'DDAH2', 'DDX39A', 'DDX46', 'DEK', 'DEPDC1', 'DEPDC1B', 'DESI2',
	'DHFR', 'DIAPH3', 'DKC1', 'DLEU2', 'DLGAP5', 'DNA2', 'DNAJB1',
	'DNAJC9', 'DNMT1', 'DPM1', 'DR1', 'DSCC1', 'DSN1', 'DTL', 'DTYMK',
	'DUSP16', 'DUT', 'DYNLL1', 'DYRK1A', 'E2F3', 'E2F7', 'E2F8',
	'ECT2', 'EED', 'EEF1D', 'EID1', 'EIF1AX', 'EIF2S2', 'EIF4A3',
	'EIF4E', 'EIF5', 'EMC9', 'ENAH', 'ENO1', 'ENY2', 'ERH', 'ESCO2',
	'EWSR1', 'EXOSC8', 'EZH2', 'FAM111B', 'FAM122B', 'FAM72C',
	'FAM72D', 'FAM83D', 'FANCB', 'FANCD2', 'FANCI', 'FANCL', 'FBL',
	'FBXL5', 'FBXO5', 'FDPS', 'FDX1', 'FEN1', 'FGFR1OP', 'FILIP1L',
	'FOXM1', 'FUS', 'FUZ', 'FXR1', 'FZR1', 'G2E3', 'G3BP1', 'GABPB1',
	'GAPDH', 'GAS2L3', 'GEMIN2', 'GEN1', 'GGCT', 'GGH', 'GINS2',
	'GLO1', 'GMNN', 'GMPS', 'GNG5', 'GPBP1', 'GPSM2', 'GTSE1', 'H1FX',
	'H2AFV', 'H2AFX', 'H2AFY', 'H2AFZ', 'HACD3', 'HADH', 'HAT1',
	'HAUS1', 'HAUS6', 'HAUS8', 'HDAC2', 'HDGF', 'HELLS', 'HES1',
	'HINT1', 'HIRIP3', 'HIST1H1A', 'HIST1H1C', 'HIST1H1D', 'HIST1H2BH',
	'HIST1H4C', 'HIST2H2AC', 'HJURP', 'HMG20B', 'HMGA1', 'HMGA2',
	'HMGB1', 'HMGB2', 'HMGB3', 'HMGN1', 'HMGN2', 'HMGN3', 'HMGN5',
	'HMGXB4', 'HMMR', 'HNRNPA0', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPA3',
	'HNRNPAB', 'HNRNPC', 'HNRNPD', 'HNRNPDL', 'HNRNPF', 'HNRNPH3',
	'HNRNPK', 'HNRNPLL', 'HNRNPM', 'HNRNPU', 'HNRNPUL1', 'HP1BP3',
	'HPF1', 'HSD17B11', 'HSP90AA1', 'HSP90B1', 'HSPA13', 'HSPA1B',
	'HSPB11', 'HSPD1', 'HSPE1', 'HYLS1', 'IDH2', 'IFT122', 'IGF2BP3',
	'IKBIP', 'ILF2', 'ILF3', 'ILVBL', 'IMMP1L', 'INCENP', 'IPO5',
	'IQGAP3', 'ISCA2', 'ISOC1', 'ITGAE', 'ITGB3BP', 'JADE1', 'JPT1',
	'KATNBL1', 'KCTD9', 'KIAA0586', 'KIF11', 'KIF14', 'KIF15',
	'KIF18A', 'KIF18B', 'KIF20A', 'KIF20B', 'KIF22', 'KIF23', 'KIF2C',
	'KIF4A', 'KIF5B', 'KIFC1', 'KMT5A', 'KNL1', 'KNSTRN', 'KPNA2',
	'KPNB1', 'LARP7', 'LBR', 'LCORL', 'LDHA', 'LDHB', 'LIG1', 'LIN52',
	'LINC01224', 'LINC01572', 'LMNB1', 'LMNB2', 'LRR1', 'LSM14A',
	'LSM2', 'LSM3', 'LSM4', 'LSM5', 'LSM6', 'LSM7', 'LSM8', 'LUC7L2',
	'MAD2L1', 'MAGI1', 'MAGOH', 'MAGOHB', 'MAPK1IP1L', 'MAPRE1',
	'MARCKS', 'MASTL', 'MBNL2', 'MCM10', 'MCM2', 'MCM3', 'MCM4',
	'MCM5', 'MCM7', 'MDH1', 'MED30', 'MELK', 'MGME1', 'MIS18A',
	'MIS18BP1', 'MKI67', 'MMS22L', 'MND1', 'MNS1', 'MORF4L2',
	'MPHOSPH9', 'MRE11', 'MRPL18', 'MRPL23', 'MRPL47', 'MRPL51',
	'MRPL57', 'MRPS34', 'MTFR2', 'MYBL2', 'MYEF2', 'MZT1', 'MZT2B',
	'NAA38', 'NAA50', 'NAE1', 'NAP1L1', 'NAP1L4', 'NASP', 'NCAPD2',
	'NCAPD3', 'NCAPG', 'NCAPG2', 'NCAPH', 'NCL', 'NDC1', 'NDC80',
	'NDE1', 'NDUFA6', 'NDUFAF3', 'NDUFS6', 'NEDD1', 'NEIL3', 'NEK2',
	'NELFE', 'NENF', 'NFATC3', 'NFYB', 'NIPBL', 'NMU', 'NONO', 'NOP56',
	'NOP58', 'NRDC', 'NSD2', 'NSMCE2', 'NSMCE4A', 'NUCKS1', 'NUDC',
	'NUDCD2', 'NUDT1', 'NUDT15', 'NUDT21', 'NUDT5', 'NUF2', 'NUP107',
	'NUP35', 'NUP37', 'NUP50', 'NUP54', 'NUSAP1', 'ODC1', 'ODF2',
	'OIP5', 'ORC6', 'PA2G4', 'PAICS', 'PAIP2', 'PAK4', 'PAPOLA',
	'PARP1', 'PARPBP', 'PAXX', 'PBK', 'PCBD2', 'PCBP2', 'PCM1', 'PCNA',
	'PCNP', 'PDS5B', 'PHF19', 'PHF5A', 'PHGDH', 'PHIP', 'PIF1',
	'PIMREG', 'PIN1', 'PKM', 'PLCB1', 'PLGRKT', 'PLIN3', 'PLK1',
	'PLK4', 'PMAIP1', 'PNISR', 'PNN', 'PNRC2', 'POC1A', 'POLD2',
	'POLD3', 'POLE2', 'POLQ', 'POLR2C', 'POLR2D', 'POLR2G', 'POLR2J',
	'POLR2K', 'POLR3K', 'PPIA', 'PPIG', 'PPIH', 'PPP1CC', 'PPP2R3C',
	'PPP2R5C', 'PPP6R3', 'PRC1', 'PRDX3', 'PRIM1', 'PRIM2', 'PRKDC',
	'PRPF38B', 'PRPSAP1', 'PRR11', 'PSIP1', 'PSMA3', 'PSMA4', 'PSMB2',
	'PSMB3', 'PSMC3', 'PSMC3IP', 'PSMD10', 'PSMD14', 'PSMG2', 'PSRC1',
	'PTBP1', 'PTGES3', 'PTMA', 'PTMS', 'PTTG1', 'PUF60', 'RAB8A',
	'RACGAP1', 'RAD21', 'RAD51AP1', 'RAD51B', 'RAD51C', 'RAN',
	'RANBP1', 'RANGAP1', 'RASSF1', 'RBBP4', 'RBBP8', 'RBL1', 'RBM17',
	'RBM39', 'RBM8A', 'RBMX', 'RCC1', 'RDX', 'REEP4', 'RFC1', 'RFC2',
	'RFC3', 'RFC4', 'RFWD3', 'RHEB', 'RMI2', 'RNASEH2B', 'RNASEH2C',
	'RNF138', 'RNF168', 'RNF26', 'RNPS1', 'RPA1', 'RPA3', 'RPL35',
	'RPL39L', 'RPLP0', 'RPLP1', 'RPLP2', 'RPN2', 'RPP30', 'RPS15',
	'RPS16', 'RPS20', 'RPS21', 'RPSA', 'RRM1', 'RSRC1', 'RSRC2',
	'RTKN2', 'RUVBL2', 'SAC3D1', 'SAE1', 'SAP18', 'SAPCD2', 'SCAF11',
	'SCLT1', 'SDHAF3', 'SELENOK', 'SEM1', 'SEPHS1', 'SEPT10', 'SEPT2',
	'SEPT7', 'SERBP1', 'SET', 'SF1', 'SF3B2', 'SFPQ', 'SGO1', 'SGO2',
	'SHCBP1', 'SINHCAF', 'SIVA1', 'SKA1', 'SKA2', 'SKA3', 'SLBP',
	'SLC20A1', 'SLC25A3', 'SLTM', 'SMC1A', 'SMC2', 'SMC3', 'SMC4',
	'SMC5', 'SMCHD1', 'SNAPC1', 'SNRNP25', 'SNRNP40', 'SNRNP70',
	'SNRPA', 'SNRPA1', 'SNRPB', 'SNRPC', 'SNRPD1', 'SNRPD2', 'SNRPD3',
	'SNRPE', 'SNRPF', 'SNRPG', 'SON', 'SPAG5', 'SPATA5', 'SPC25',
	'SPCS2', 'SPDL1', 'SREK1', 'SRI', 'SRP9', 'SRRM1', 'SRSF1',
	'SRSF10', 'SRSF11', 'SRSF2', 'SRSF3', 'SRSF4', 'SRSF7', 'SSB',
	'SSBP1', 'SSNA1', 'SSRP1', 'ST13', 'STAG1', 'STIL', 'STIP1',
	'STK17B', 'STK3', 'STOML2', 'SUGP2', 'SUMO1', 'SUMO3', 'SUPT16H',
	'SUV39H2', 'SUZ12', 'SYNE2', 'TACC3', 'TBC1D31', 'TBC1D5', 'TDP1',
	'TEAD1', 'TEX30', 'TFDP1', 'THRAP3', 'TICRR', 'TIMELESS', 'TIMM10',
	'TK1', 'TMED5', 'TMEM106C', 'TMEM237', 'TMEM60', 'TMEM97', 'TMPO',
	'TMSB15A', 'TOP1', 'TOP2A', 'TPI1', 'TPR', 'TPRKB', 'TPX2',
	'TRA2B', 'TRAIP', 'TROAP', 'TTC28', 'TTF2', 'TTK', 'TUBA1B',
	'TUBA1C', 'TUBB', 'TUBB4B', 'TUBG1', 'TUBGCP3', 'TXNDC12', 'TYMS',
	'UBA2', 'UBB', 'UBE2C', 'UBE2D2', 'UBE2D3', 'UBE2I', 'UBE2N',
	'UBE2S', 'UBE2T', 'UHRF1', 'UNG', 'UQCC2', 'UQCC3', 'UQCRC1',
	'UQCRFS1', 'USP1', 'VBP1', 'VDAC3', 'VEZF1', 'VRK1', 'WAPL',
	'WDHD1', 'WDPCP', 'WDR34', 'WDR76', 'XPO1', 'XRCC4', 'XRCC5',
	'XRCC6', 'YAP1', 'YBX1', 'YEATS4', 'Z94721.1', 'ZFP36L1', 'ZGRF1',
	'ZMYM1', 'ZNF22', 'ZNF367', 'ZNF43', 'ZNF704', 'ZNF83', 'ZRANB3',
	'ZSCAN16-AS1', 'ZWINT'], dtype=object)

cc_genes_mouse = np.array([x[0] + x[1:].lower() for x in cc_genes_human], dtype=object)


class Cytograph2:
	def __init__(self, *, n_genes: int = 2000, n_factors: int = 64, k: int = 50, k_pooling: int = 5, outliers: bool = False, required_genes: List[str] = None, mask_cell_cycle: bool = False, feature_selection_method: str = "markers", use_poisson_pooling: bool = True) -> None:
		"""
		Run cytograph2

		Args:
			n_genes							Number of genes to select
			n_factors						Number of HPF factors
			k								Number of neighbors for KNN graph
			k_pooling						Number of neighbors for Poisson pooling
			outliers						Allow outliers and mark them
			required_genes					List of genes that must be included in any feature selection (except "cellcycle")
			mask_cell_cycle					Remove cell cycle genes (including from required_genes), unless feature_selection_method == "cellcycle"
			feature_selection_method 		"markers", "variance" or "cellcycle"
			use_poisson_pooling				If true and pooling layers exist, use them
		"""
		self.n_genes = n_genes
		self.n_factors = n_factors
		self.k_pooling = k_pooling
		self.k = k
		self.outliers = outliers
		self.required_genes = required_genes
		self.mask_cell_cycle = mask_cell_cycle
		self.feature_selection_method = feature_selection_method
		self.use_poisson_pooling = use_poisson_pooling

	def poisson_pooling(self, ds: loompy.LoomConnection) -> None:
		n_samples = ds.shape[1]
		logging.info(f"Selecting {self.n_genes} genes")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		self.genes = genes
		data = ds.sparse(rows=genes).T

		# Subsample to lowest number of UMIs
		# TODO: figure out how to do this without making the data matrix dense
		if "TotalRNA" not in ds.ca:
			(ds.ca.TotalRNA, ) = ds.map([np.sum], axis=1)
		totals = ds.ca.TotalRNA
		min_umis = np.min(totals)
		logging.info(f"Subsampling to {min_umis} UMIs")
		temp = data.toarray()
		for c in range(temp.shape[0]):
			temp[c, :] = np.random.binomial(temp[c, :].astype('int32'), min_umis / totals[c])
		data = sparse.coo_matrix(temp)

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T  # Normalize so the sums are one because JSD requires it

		if "Batch" in ds.ca and "Replicate" in ds.ca:
			technical = identify_technical_factors(theta, ds.ca.Batch, ds.ca.Replicate)
			logging.info(f"Removing {technical.sum()} technical factors")
			theta = theta[:, ~technical]
		else:
			logging.warn("Could not analyze technical factors because attributes 'Batch' and 'Replicate' are missing")

		# KNN in HPF space
		logging.info(f"Computing KNN (k={self.k_pooling}) in latent space")
		nn = NNDescent(data=theta, metric=jensen_shannon_distance)
		indices, distances = nn.query(theta, k=self.k_pooling)
		# Note: we convert distances to similarities here, to support Poisson smoothing below
		knn = sparse.csr_matrix(
			(1 - np.ravel(distances), np.ravel(indices), np.arange(0, distances.shape[0] * distances.shape[1] + 1, distances.shape[1])), 		(theta.shape[0], theta.shape[0])
		)
		knn.setdiag(1)

		# Poisson pooling
		logging.info(f"Poisson pooling")
		ds["pooled"] = 'int32'
		if "spliced" in ds.layers:
			ds["spliced_pooled"] = 'int32'
			ds["unspliced_pooled"] = 'int32'
		for (ix, indexes, view) in ds.scan(axis=0):
			if "spliced" in ds.layers:
				ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["spliced"][:, :].T).T
				ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view.layers["unspliced"][:, :].T).T
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = ds["spliced_pooled"][indexes.min(): indexes.max() + 1, :] + ds["unspliced_pooled"][indexes.min(): indexes.max() + 1, :]
			else:
				ds["pooled"][indexes.min(): indexes.max() + 1, :] = knn.dot(view[:, :].T).T

	def feature_selection_by_cell_cycle(self, ds: loompy.LoomConnection, main_layer: str) -> np.ndarray:
		cc_genes = cc_genes_human if species(ds) == "Homo sapiens" else cc_genes_mouse
		genes = np.where(np.isin(ds.ra.Gene, cc_genes))[0]
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		return genes

	def feature_selection_by_variance(self, ds: loompy.LoomConnection, main_layer: str) -> np.ndarray:
		cc_genes = cc_genes_human if species(ds) == "Homo sapiens" else cc_genes_mouse
		normalizer = cg.Normalizer(False, layer=main_layer)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes, layer=main_layer).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		# Make sure to include these genes
		genes = np.union1d(genes, np.where(np.isin(ds.ra.Gene, self.required_genes))[0])
		# Mask cell cycle
		if self.mask_cell_cycle:
			genes = np.setdiff1d(genes, np.where(np.isin(ds.ra.Gene, cc_genes))[0])
		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		return genes

	def feature_selection_by_markers(self, ds: loompy.LoomConnection, main_layer: str) -> np.ndarray:
		cc_genes = cc_genes_human if species(ds) == "Homo sapiens" else cc_genes_mouse

		logging.info("Selecting up to %d marker genes", self.n_genes)
		normalizer = cg.Normalizer(False, layer=main_layer)
		normalizer.fit(ds)
		genes = cg.FeatureSelection(self.n_genes, layer=main_layer).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
		n_cells = ds.shape[1]
		n_components = min(50, n_cells)
		logging.info("PCA projection to %d components", n_components)
		pca = cg.PCAProjection(genes, max_n_components=n_components, layer=main_layer)
		pca_transformed = pca.fit_transform(ds, normalizer)
		transformed = pca_transformed

		logging.info("Generating balanced KNN graph")
		np.random.seed(0)
		k = min(self.k, n_cells - 1)
		bnn = cg.BalancedKNN(k=k, maxl=2 * k, sight_k=2 * k)
		bnn.fit(transformed)
		knn = bnn.kneighbors_graph(mode='connectivity')
		knn = knn.tocoo()
		mknn = knn.minimum(knn.transpose()).tocoo()

		logging.info("MKNN-Louvain clustering with outliers")
		(a, b, w) = (mknn.row, mknn.col, mknn.data)
		lj = cg.LouvainJaccard(resolution=1, jaccard=False)
		labels = lj.fit_predict(knn)
		bigs = np.where(np.bincount(labels) >= 10)[0]
		mapping = {k: v for v, k in enumerate(bigs)}
		labels = np.array([mapping[x] if x in bigs else -1 for x in labels])

		n_labels = np.max(labels) + 1
		logging.info("Found " + str(n_labels) + " preliminary clusters")

		logging.info("Marker selection")
		temp = None
		if "Clusters" in ds.ca:
			temp = ds.ca.Clusters
		ds.ca.Clusters = labels - labels.min()
		(genes, _, _) = cg.MarkerSelection(n_markers=int(500 / n_labels), findq=False).fit(ds)
		if temp is not None:
			ds.ca.Clusters = temp

		# Make sure to include these genes
		genes = np.union1d(genes, np.where(np.isin(ds.ra.Gene, self.required_genes))[0])
		# Mask cell cycle
		if self.mask_cell_cycle:
			genes = np.setdiff1d(genes, np.where(np.isin(ds.ra.Gene, cc_genes))[0])

		selected = np.zeros(ds.shape[0])
		selected[genes] = 1
		ds.ra.Selected = selected
		return genes

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info(f"Running cytograph on {ds.shape[1]} cells")
		n_samples = ds.shape[1]

		if self.use_poisson_pooling and ("pooled" in ds.layers):
			main_layer = "pooled"
			spliced_layer = "spliced_pooled"
			unspliced_layer = "unspliced_pooled"
		else:
			main_layer = ""
			spliced_layer = "spliced"
			unspliced_layer = "unspliced"
		# Select genes
		logging.info(f"Selecting {self.n_genes} genes")
		if self.feature_selection_method == "variance":
			genes = self.feature_selection_by_variance(ds, main_layer)
		elif self.feature_selection_method == "markers":
			genes = self.feature_selection_by_markers(ds, main_layer)
		elif self.feature_selection_method == "cellcycle":
			genes = self.feature_selection_by_cell_cycle(ds, main_layer)

		# Load the data for the selected genes
		data = ds[main_layer].sparse(rows=genes).T

		# HPF factorization
		logging.info(f"HPF to {self.n_factors} latent factors")
		hpf = cg.HPF(k=self.n_factors, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		beta_all = np.zeros((ds.shape[0], hpf.beta.shape[1]))
		beta_all[genes] = hpf.beta
		# Save the unnormalized factors
		ds.ra.HPF_beta = beta_all
		ds.ca.HPF_theta = hpf.theta
		# Here we normalize so the sums over components are one, because JSD requires it
		# and because otherwise the components will be exactly proportional to cell size
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T
		beta = (hpf.beta.T / hpf.beta.sum(axis=1)).T
		beta_all[genes] = beta
		if "Batch" in ds.ca and "Replicate" in ds.ca:
			technical = identify_technical_factors(theta, ds.ca.Batch, ds.ca.Replicate)
			logging.info(f"Removing {technical.sum()} technical factors")
			theta = theta[:, ~technical]
			beta = beta[:, ~technical]
			beta_all = beta_all[:, ~technical]
		else:
			logging.warn("Could not analyze technical factors because attributes 'Batch' and 'Replicate' are missing")
		# Save the normalized factors
		ds.ra.HPF = beta_all
		ds.ca.HPF = theta

		# HPF factorization of spliced/unspliced
		if "spliced" in ds.layers:
			logging.info(f"HPF of spliced molecules")
			data_spliced = ds[spliced_layer].sparse(rows=genes).T
			theta_spliced = hpf.transform(data_spliced)
			theta_spliced = (theta_spliced.T / theta_spliced.sum(axis=1)).T
			if "Batch" in ds.ca and "Replicate" in ds.ca:
				theta_spliced = theta_spliced[:, ~technical]
			ds.ca.HPF_spliced = theta_spliced
			logging.info(f"HPF of unspliced molecules")
			data_unspliced = ds[unspliced_layer].sparse(rows=genes).T
			theta_unspliced = hpf.transform(data_unspliced)
			theta_unspliced = (theta_unspliced.T / theta_unspliced.sum(axis=1)).T
			if "Batch" in ds.ca and "Replicate" in ds.ca:
				theta_unspliced = theta_unspliced[:, ~technical]
			ds.ca.HPF_unspliced = theta_unspliced

		# Expected values
		logging.info(f"Computing expected values")
		ds["expected"] = 'float32'  # Create a layer of floats
		log_posterior_proba = np.zeros(n_samples)
		theta_unnormalized = hpf.theta[:, ~technical] if "Batch" in ds.ca else hpf.theta
		data = data.toarray()
		start = 0
		batch_size = 6400
		if "spliced" in ds.layers:
			ds["spliced_exp"] = 'float32'
			ds['unspliced_exp'] = 'float32'
		while start < n_samples:
			# Compute PPV (using normalized theta)
			ds["expected"][:, start: start + batch_size] = beta_all @ theta[start: start + batch_size, :].T
			# Compute PPV using raw theta, for calculating posterior probability of the observations
			ppv_unnormalized = beta @ theta_unnormalized[start: start + batch_size, :].T
			log_posterior_proba[start: start + batch_size] = poisson.logpmf(data.T[:, start: start + batch_size], ppv_unnormalized).sum(axis=0)
			if "spliced" in ds.layers:
				ds["spliced_exp"][:, start: start + batch_size] = beta_all @ theta_spliced[start: start + batch_size, :].T
				ds["unspliced_exp"][:, start: start + batch_size] = beta_all @ theta_unspliced[start: start + batch_size, :].T
			start += batch_size
		ds.ca.HPF_LogPP = log_posterior_proba

		# logging.info(f"Computing balanced KNN (k = {self.k}) in latent space")
		bnn = cg.BalancedKNN(k=self.k, metric="js", maxl=2 * self.k, sight_k=2 * self.k, n_jobs=-1)
		bnn.fit(theta)
		knn = bnn.kneighbors_graph(mode='distance')
		knn.eliminate_zeros()
		mknn = knn.minimum(knn.transpose())
		# Convert distances to similarities
		knn.data = 1 - knn.data
		mknn.data = 1 - mknn.data
		ds.col_graphs.KNN = knn
		ds.col_graphs.MKNN = mknn
		# Compute the effective resolution
		d = 1 - knn.data
		d = d[d < 1]
		radius = np.percentile(d, 90)
		logging.info(f"90th percentile radius: {radius:.02}")
		ds.attrs.radius = radius
		knn.setdiag(0)
		knn = knn.tocoo()
		inside = knn.data > 1 - radius
		rnn = sparse.coo_matrix((knn.data[inside], (knn.row[inside], knn.col[inside])), shape=knn.shape)
		ds.col_graphs.RNN = rnn

		logging.info(f"2D tSNE embedding from latent space")
		ds.ca.TSNE = tsne(theta, metric="js", radius=radius)

		logging.info(f"2D UMAP embedding from latent space")
		ds.ca.UMAP = UMAP(n_components=2, metric=jensen_shannon_distance, n_neighbors=self.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

		logging.info(f"3D UMAP embedding from latent space")
		ds.ca.UMAP3D = UMAP(n_components=3, metric=jensen_shannon_distance, n_neighbors=self.k // 2, learning_rate=0.3, min_dist=0.25).fit_transform(theta)

		logging.info("Clustering by polished Louvain")
		pl = cg.PolishedLouvain(outliers=self.outliers)
		labels = pl.fit_predict(ds, graph="RNN", embedding="UMAP3D")
		ds.ca.Clusters = labels + min(labels)
		ds.ca.Outliers = (labels == -1).astype('int')
		logging.info(f"Found {labels.max() + 1} clusters")

		if "spliced" in ds.layers:
			logging.info("Fitting gamma for velocity inference")
			selected = ds.ra.Selected == 1
			n_genes = ds.shape[0]
			s = ds["spliced_exp"][selected, :]
			u = ds["unspliced_exp"][selected, :]
			gamma, _ = fit_gamma(s, u)
			gamma_all = np.zeros(n_genes)
			gamma_all[selected] = gamma
			ds.ra.Gamma = gamma_all

			logging.info("Computing velocity")
			velocity = u - gamma[:, None] * s
			ds["velocity"] = "float32"
			ds["velocity"][selected, :] = velocity

			logging.info("Projecting velocity to latent space")
			beta = ds.ra.HPF
			ds.ca.HPFVelocity = (beta[ds.ra.Selected == 1].T @ velocity).T.astype("float32")

			logging.info("Projecting velocity to TSNE 2D embedding")
			ve = VelocityEmbedding(data_source="HPF", velocity_source="HPFVelocity", embedding_name="TSNE", neighborhood_type="RNN", points_kind="cells", min_neighbors=0)
			ds.ca.TSNEVelocity = ve.fit(ds)
			# Embed velocity on a 50x50 grid
			ve = VelocityEmbedding(data_source="HPF", velocity_source="HPFVelocity", embedding_name="TSNE", neighborhood_type="radius", neighborhood_size=5, points_kind="grid", num_points=50, min_neighbors=5)
			ds.attrs.TSNEVelocity = ve.fit(ds)
			ds.attrs.TSNEVelocityPoints = ve.points

			logging.info("Projecting velocity to UMAP 2D embedding")
			ve = VelocityEmbedding(data_source="HPF", velocity_source="HPFVelocity", embedding_name="UMAP", neighborhood_type="RNN", points_kind="cells", min_neighbors=0)
			ds.ca.UMAPVelocity = ve.fit(ds)
			# Embed velocity on a 50x50 grid
			ve = VelocityEmbedding(data_source="HPF", velocity_source="HPFVelocity", embedding_name="UMAP", neighborhood_type="radius", neighborhood_size=0.5, points_kind="grid", num_points=50, min_neighbors=5)
			ds.attrs.UMAPVelocity = ve.fit(ds)
			ds.attrs.UMAPVelocityPoints = ve.points

		logging.info("Inferring cell cycle")
		cca = CellCycleAnnotator(ds)
		cca.annotate_loom()
