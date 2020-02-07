import numpy as np
import loompy
from .heatmap import Heatmap
from cytograph.species import Species


def TF_heatmap(ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, out_file: str = None, layer: str = "pooled") -> None:
	TFs = Species.detect(ds).genes.TFs
	enrichment = dsagg["enrichment"][:, :]
	enrichment = enrichment[np.isin(dsagg.ra.Gene, TFs), :]
	genes = dsagg.ra.Gene[np.isin(dsagg.ra.Gene, TFs)]
	genes = genes[np.argsort(-enrichment, axis=0)[:10, :]].T  # (n_clusters, n_genes)
	genes = np.unique(genes)  # 1d array of unique genes, sorted

	hm = Heatmap(np.isin(ds.ra.Gene, genes), attrs={
		"Clusters": "categorical",
		"SampleName": "categorical",
		"SampleID": "categorical",
		"Tissue": "ticker",
		"Sex": "categorical",
		"Age": "categorical",
		"TotalUMI": "plasma:log",
		"CellCycle_G1": "viridis:log",
		"CellCycle_S": "viridis:log",
		"CellCycle_G2M": "viridis:log",
		"DoubletFinderScore": "viridis",
		"DoubletFinderFlag": "PiYG_r",
	}, layer=layer)
	hm.plot(ds, dsagg, out_file=out_file)
