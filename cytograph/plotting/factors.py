import matplotlib.pyplot as plt
import numpy as np

import loompy


def factors(ds: loompy.LoomConnection, base_name: str) -> None:
	offset = 0
	theta = ds.ca.HPF
	beta = ds.ra.HPF
	n_factors = theta.shape[1]
	while offset < n_factors:
		fig = plt.figure(figsize=(15, 15))
		fig.subplots_adjust(hspace=0, wspace=0)
		for nnc in range(offset, offset + 16):
			if nnc >= n_factors:
				break
			ax = plt.subplot(4, 4, (nnc - offset) + 1)
			plt.xticks(())
			plt.yticks(())
			plt.axis("off")
			plt.scatter(x=ds.ca.TSNE[:, 0], y=ds.ca.TSNE[:, 1], c='lightgrey', marker='.', alpha=0.5, s=60, lw=0)
			cells = theta[:, nnc] > np.percentile(theta[:, nnc], 99) * 0.1
			cmap = "viridis"
			plt.scatter(x=ds.ca.TSNE[cells, 0], y=ds.ca.TSNE[cells, 1], c=theta[:, nnc][cells], marker='.', alpha=0.5, s=60, cmap=cmap, lw=0)
			ax.text(.01, .99, '\n'.join(ds.ra.Gene[np.argsort(-beta[:, nnc])][:9]), horizontalalignment='left', verticalalignment="top", transform=ax.transAxes)
			ax.text(.99, .9, f"{nnc}", horizontalalignment='right', transform=ax.transAxes, fontsize=12)
		plt.savefig(base_name + f"{offset}.png", dpi=144)
		offset += 16
		plt.close()
