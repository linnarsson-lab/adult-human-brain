import numpy as np
import loompy
from .heatmap import Heatmap


def markerheatmap(ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, out_file: str = "") -> None:
	n_clusters = ds.ca.Clusters.max() + 1
	hm = Heatmap(np.arange(10 * n_clusters), attrs={
		"SampleName": "categorical",
		"SampleID": "categorical",
		"Tissue": "ticker",
		"Sex": "categorical",
		"Age": "categorical",
		"TotalUMI": "plasma:log",
		"CellCycle_G1": "viridis:log",
		"CellCycle_S": "viridis:log",
		"CellCycle_G2M": "viridis:log",
		"ScrubletScore": "viridis",
		"ScrubletFlag": "PiYG_r",
		"Clusters": "categorical"
	})
	hm.plot(ds, dsagg, out_file=out_file)
