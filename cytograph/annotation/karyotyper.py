import loompy
import numpy as np

chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"]


class Karyotyper:
	def __init__(self, window: int = 200, use_chr: bool = True) -> None:
		self.window = window
		if use_chr:
			self.chromosomes = chromosomes
		else:
			self.chromosomes = [c[3:] for c in chromosomes]

	def fit(self, refpath: str) -> None:
		with loompy.connect(refpath) as ds:
			self.median = np.median(ds.ca.TotalUMIs)
			size_f = self.median / ds.ca.TotalUMIs
			# Count the number of windows
			self.n_bins = 0
			for c in chromosomes:
				genes = (ds.ra.Chromosome == c)
				self.n_bins += np.ceil((ds.ra.Chromosome == c).sum() / self.window)
			self.n_bins = int(self.n_bins)
			self.ref_profile = np.zeros(self.n_bins)
			self.chromosomes = np.zeros(self.n_bins, dtype=object)
			for ix, selection, view in ds.scan(axis=1):
				offset = int(0)
				for c in chromosomes:
					genes = (ds.ra.Chromosome == c)
					ordering = np.argsort(ds.ra.ChromosomeStart[genes])
					for i in range(0, genes.sum(), self.window):
						self.ref_profile[(i // self.window) + offset] += (view[genes, :][ordering, :][i: i + self.window, :].sum(axis=0) * size_f[selection]).sum()
						self.chromosomes[(i // self.window) + offset] = c
					offset += int(np.ceil((ds.ra.Chromosome == c).sum() / self.window))
			self.ref_profile /= ds.shape[1]

	def transform(self, testpath: str) -> np.ndarray:
		with loompy.connect(testpath) as ds:
			size_f = self.median / ds.ca.TotalUMIs
			self.test_profile = np.zeros((self.n_bins, ds.shape[1]))
			for ix, selection, view in ds.scan(axis=1):
				offset = int(0)
				for c in chromosomes:
					genes = (ds.ra.Chromosome == c)
					ordering = np.argsort(ds.ra.ChromosomeStart[genes])
					for i in range(0, genes.sum(), self.window):
						self.test_profile[(i // self.window) + offset, selection] = view[genes, :][ordering, :][i: i + self.window].sum(axis=0) * size_f[selection]
					offset += int(np.ceil((ds.ra.Chromosome == c).sum() / self.window))
		return self.test_profile
