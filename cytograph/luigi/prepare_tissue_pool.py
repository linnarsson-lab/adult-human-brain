from typing import *
import os
import csv
import numpy as np
import pickle
import logging
import luigi
import cytograph as cg
import loompy


class PrepareTissuePool(luigi.Task):
	"""
	Luigi Task to prepare tissue-level files from raw sample files, including gene and cell validation
	"""
	tissue = luigi.Parameter()

	def requires(self) -> List[luigi.Task]:
		samples = cg.PoolSpec().samples_for_tissue(self.tissue)
		return [cg.TrainClassifier()] + [cg.Sample(sample=s) for s in samples]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.tissue + ".loom"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			attrs = {"title": self.tissue}
			valid_cells = []
			sample_files = [s.fn for s in self.input()[1:]]
			for sample in sample_files:
				# Connect and perform file-specific cell validation
				logging.info("Marking invalid cells")
				ds = loompy.connect(sample)
				(mols, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
				valid_cells.append(np.logical_and(mols >= 600, (mols / genes) >= 1.2).astype('int'))
				ds.close()

			logging.info("Creating combined loom file")
			loompy.combine(sample_files, out_file, key="Accession", file_attrs=attrs)

			# Validating genes
			logging.info("Marking invalid genes")
			ds = loompy.connect(out_file)
			nnz = ds.map(np.count_nonzero, axis=0)
			valid_genes = np.logical_and(nnz > 20, nnz < ds.shape[1] * 0.6)
			ds.set_attr("_Valid", valid_genes, axis=0)
			ds.set_attr("_Valid", np.concatenate(valid_cells), axis=1)
			logging.info("Classifying cells by major class")
			with open(self.input()[0].fn, "rb") as f:
				clf = pickle.load(f)
			(probs, labels) = clf.predict_proba(ds)
			labels = np.array([x.replace("-", "_") for x in labels])
			ds.set_attr("Class", labels[np.argmax(probs, axis=1)], axis=1)
			for ix, label in enumerate(labels):
				ds.set_attr("Class_" + label, probs[:, ix], axis=1)
			ds.close()
