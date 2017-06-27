from typing import *
import os
import loompy
import numpy as np
import cytograph as cg
import luigi


class MarkerEnrichment:
	"""
	DEPRECATED

	Use the MarkerSelection class instead for a more powerful enrichment calculation, which avoids
	division by zero and properly accounts for low counts and averages.
	"""
	def __init__(self, power: float) -> None:
		self.enrichment = None  # type: np.ndarray
		self.power = power
		self.genes = None  # type: np.ndarray
		self.valid = None  # type: np.ndarray

	def fit(self, ds: loompy.LoomConnection) -> None:
		cells = np.where(ds.col_attrs["Clusters"] >= 0)[0]
		labels = ds.col_attrs["Clusters"][cells]
		n_labels = np.max(labels) + 1
		scores1 = np.empty((ds.shape[0], n_labels))
		scores2 = np.empty((ds.shape[0], n_labels))

		j = 0
		for (ix, selection, vals) in ds.batch_scan(cells=cells, genes=None, axis=0, batch_size=cg.memory().axis0):
			for j, row in enumerate(selection):
				data = vals[j, :]
				mu0 = np.mean(data)
				f0 = np.count_nonzero(data)
				score1 = np.zeros(n_labels)
				score2 = np.zeros(n_labels)
				for lbl in range(n_labels):
					if np.sum(labels == lbl) == 0:
						continue
					sel = data[np.where(labels == lbl)[0]]
					if mu0 == 0 or f0 == 0:
						score1[lbl] = 0
						score2[lbl] = 0
					else:
						score1[lbl] = np.mean(sel) / mu0
						score2[lbl] = np.count_nonzero(sel)
				scores1[row, :] = score1
				scores2[row, :] = score2
		self.enrichment = scores1 * np.power(scores2, self.power)
		self.genes = ds.Gene
		self.valid = ds.row_attrs["_Valid"]

	def save(self, fname: str) -> None:
		with open(fname, "w") as f:
			f.write("Gene\t")
			f.write("Valid\t")
			for ix in range(self.enrichment.shape[1]):
				f.write(str(ix) + "\t")
			f.write("\n")

			for row in range(self.enrichment.shape[0]):
				f.write(self.genes[row] + "\t")
				f.write(str(self.valid[row]) + "\t")
				for ix in range(self.enrichment.shape[1]):
					f.write(str(self.enrichment[row, ix]) + "\t")
				f.write("\n")