from typing import *
import os
import csv
import numpy as np
import pickle
import logging
import luigi
import cytograph as cg
import loompy
import numpy.core.defchararray as npstr


class PrepareTissuePool(luigi.Task):
	"""
	Luigi Task to prepare tissue-level files from raw sample files, including gene and cell validation
	"""
	tissue = luigi.Parameter()

	def requires(self) -> List[luigi.Task]:
		samples = cg.PoolSpec().samples_for_tissue(self.tissue)
		return [cg.TrainClassifier()] + [cg.Sample(sample=s) for s in samples]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L0_" + self.tissue + ".loom"))

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

				logging.info("Computing mito/ribo ratio")
				try:
					mito = np.where(npstr.startswith(ds.row_attrs["Gene"], "mt-"))[0]
					ribo = np.where(npstr.startswith(ds.row_attrs["Gene"], "Rpl"))[0]
					ribo = np.union1d(ribo, np.where(npstr.startswith(ds.row_attrs["Gene"], "Rps"))[0])
					if (len(ribo) == 0) or (len(mito) == 0):
						# I raise this kind of error becouse is the same it would be raised if this happen
						raise UnboundLocalError
					mitox = ds[mito, :]
					ribox = ds[ribo, :]
					ratio = (mitox.sum(axis=0) + 1) / (ribox.sum(axis=0) + 1)
					ds.set_attr("MitoRiboRatio", ratio, axis=1)
				except UnboundLocalError:
					pass
				ds.close()

			logging.info("Creating combined loom file")
			loompy.combine(sample_files, out_file, key="Accession", file_attrs=attrs)

			# Validating genes
			logging.info("Marking invalid genes")
			ds = loompy.connect(out_file)
			nnz = ds.map([np.count_nonzero], axis=0)[0]
			valid_genes = np.logical_and(nnz > 20, nnz < ds.shape[1] * 0.6)
			ds.set_attr("_Valid", valid_genes, axis=0)

			logging.info("Marking invalid cells")
			ds.set_attr("_Valid", np.concatenate(valid_cells), axis=1)
			n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
			n_total = ds.shape[1]
			logging.info("%d of %d cells were valid", n_valid, n_total)

			# TODO : change the luigi pipeline so that is more general and the exception below is not needed
			if os.path.exists(os.path.join(cg.paths().build, "classifier.pickle")):
				logging.info("Classifying cells by major class")
				with open(self.input()[0].fn, "rb") as f:
					clf = pickle.load(f)  # type: cg.Classifier
				(probs, labels, classes) = clf.predict_proba(ds)
				mapping = {
					"Astrocyte": "Astrocyte",
					"Ependymal": "Astrocyte",
					"Neurons": "Neurons",
					"Oligos": "Oligos",
					"Cycling": "Cycling",
					"Immune": "Immune",
					"Vascular": "Vascular",
					"OEC": "Astrocyte",
					"Schwann": "Oligos",
					"Excluded": "Excluded",
					"Unknown": "Excluded"
				}
				classes = np.array(classes, dtype=np.object_)
				classes_pooled = np.array([mapping[c] for c in classes], dtype=np.object_)

				# add erythrocytes
				hbb = np.where(ds.Gene == "Hbb-bs")[0][0]
				ery = np.where(ds[hbb, :] > 2)[0]
				classes[ery] = "Erythrocyte"
				classes_pooled[ery] = "Erythrocyte"
				# mask invalid cells
				classes[ds.col_attrs["_Valid"] == 0] = "Excluded"
				classes_pooled[ds.col_attrs["_Valid"] == 0] = "Excluded"
				ds.set_attr("Class", classes_pooled.astype('str'), axis=1)
				ds.set_attr("Class0", classes.astype('str'), axis=1)
				for ix, label in enumerate(labels):
					ds.set_attr("Class_" + label, probs[:, ix], axis=1)
			else:
				logging.info("Classification cannot be performed on this dataset (no classifier found)")
				ds.set_attr("Class", ["Excluded"] * ds.shape[1], axis=1)
				ds.set_attr("Class0", ["Unknown"] * ds.shape[1], axis=1)
			ds.close()
