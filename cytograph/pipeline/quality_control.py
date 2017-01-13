import os
from typing import *
import logging
from shutil import copyfile
import numpy as np
import loompy
import differentiation_topology as dt
import luigi


class QualityControl(luigi.Task):
	"""
	Luigi Task to copy a Loom file into the build folder, perform QC on cells, and estimate doublets
	"""
	source_folder = luigi.Parameter(default="")
	build_folder = luigi.Parameter(default="")
	sample = luigi.Parameter()

	def output(self) -> luigi.LocalTarget:
		return luigi.LocalTarget(os.path.join(self.build_folder, "%s.loom" % self.sample))

	def requires(self) -> List[Any]:
		return []

	def run(self) -> None:
		logging.info("QC: " + self.sample)
		fname = os.path.join(self.build_folder, "%s.loom" % self.sample)
		copyfile(os.path.join(self.source_folder, self.sample + ".loom"), fname)

		# Connect and perform file-specific QC and validation
		ds = loompy.connect(fname)

		# Validate cells
		(mols, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
		valid = np.logical_and(np.logical_and(mols >= 600, (mols / genes) >= 1.2), np.logical_and(mols <= 20000, genes >= 500)).astype('int')
		ds.set_attr("_Valid", valid, axis=1)

		# Estimate doublets
		mog_ix = np.where(ds.Gene == "Mog")[0]
		stmn3_ix = np.where(ds.Gene == "Stmn3")[0]
		n_total = valid.sum()
		n_mog = np.logical_and(ds[mog_ix, valid] > 0, ds[stmn3_ix, valid] == 0).sum()
		n_stmn3 = np.logical_and(ds[stmn3_ix, valid] > 0, ds[mog_ix, valid] == 0).sum()
		n_both = np.logical_and(ds[stmn3_ix, valid] > 0, ds[mog_ix, valid] > 0).sum()
		f_doublets = n_total * n_both / (n_both + n_stmn3) / (n_both + n_mog)
		ds.set_attr("f_doublets", np.ones(ds.shape[1]) * f_doublets, axis=1)
		ds.close()
