from typing import *
import os
import csv
import numpy as np
import pandas as pd
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
		return [cg.Sample(sample=s) for s in samples]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L0_" + self.tissue + ".loom"))

	def run(self) -> None:
		# Load metadata
		metadata: np.ndarray = None
		meta_attrs: np.ndarray = None
		metadata_file = os.path.join(cg.paths().samples, "metadata", "metadata.xlsx")
		if os.path.exists(metadata_file):
			temp = pd.read_excel(metadata_file)
			meta_attrs = temp.columns.values
			metadata = temp.values

		with self.output().temporary_path() as out_file:
			attrs = {"title": self.tissue}
			valid_cells = []
			sample_files = [s.fn for s in self.input()]
			for sample in sample_files:
				# Connect and perform file-specific cell validation
				ds = loompy.connect(sample)

				if metadata is not None:
					logging.info("Inserting metadata")
					vals = temp.values[metadata[:, 0] == os.path.basename(sample)][0]
					for ix in range(vals.shape[0]):
						ds.set_attr(meta_attrs[ix], np.array([vals[ix]] * ds.shape[1]), axis=1)

				logging.info("Marking invalid cells")
				(mols, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
				valid_cells.append(np.logical_and(mols >= 600, (mols / genes) >= 1.2).astype('int'))
				ds.set_attr("_Total", mols, axis=1)
				ds.set_attr("_NGenes", genes, axis=1)
				
				logging.info("Computing mito/ribo ratio for " + sample)
				mito = np.where(npstr.startswith(ds.row_attrs["Gene"], "mt-"))[0]
				ribo = np.where(npstr.startswith(ds.row_attrs["Gene"], "Rpl"))[0]
				ribo = np.union1d(ribo, np.where(npstr.startswith(ds.row_attrs["Gene"], "Rps"))[0])
				if len(ribo) > 0 and len(mito) > 0:
					mitox = ds[mito, :]
					ribox = ds[ribo, :]
					ratio = (mitox.sum(axis=0) + 1) / (ribox.sum(axis=0) + 1)
					ds.set_attr("MitoRiboRatio", ratio, axis=1)
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
			
			classifier_path = os.path.join(cg.paths().build, "classifier.pickle")
			if os.path.exists(classifier_path):
				logging.info("Classifying cells by major class")
				with open(classifier_path, "rb") as f:
					clf = pickle.load(f)  # type: cg.Classifier
				(classes, probs, class_labels) = clf.predict(ds, probability=True)

				mapping = {
					"Astrocyte": "AstroEpendymal",
					"Astrocyte,Cycling": "AstroEpendymal",
					"Astrocyte,Immune": None,
					"Astrocyte,Neurons": None,
					"Astrocyte,Oligos": None,
					"Astrocyte,Vascular": None,
					"Bergmann-glia": "AstroEpendymal",
					"Blood": "Blood",
					"Blood,Cycling": "Blood",
					"Blood,Vascular": None,
					"Ependymal": "AstroEpendymal",
					"Ex-Astrocyte": None,
					"Ex-Blood": None,
					"Ex-Immune": None,
					"Ex-Neurons": None,
					"Ex-Oligos": None,
					"Ex-Vascular": None,
					"Immune": "Immune",
					"Immune,Neurons": None,
					"Immune,Oligos": None,
					"Neurons": "Neurons",
					"Neurons,Cycling": "Neurons",
					"Neurons,Immune": None,
					"Neurons,Oligos": None,
					"Neurons,Satellite-glia": None,
					"OEC": "AstroEpendymal",
					"Oligos": "Oligos",
					"Oligos,Cycling": "Oligos",
					"Immune,Oligos": None,
					"Oligos,Vascular": None,
					"Satellite-glia": "PeripheralGlia",
					"Satellite-glia,Cycling": "PeripheralGlia",
					"Schwann": "PeripheralGlia",
					"Satellite-glia,Schwann": None,
					"Ttr": "AstroEpendymal",
					"Vascular": "Vascular",
					"Vascular,Cycling": "Vascular",
					"Neurons,Vascular": None,
					"Vascular,Oligos": None,
					"Satellite-glia,Vascular": None,
					"Unknown": None
				}

				classes_pooled = np.array([str(mapping[c]) for c in classes], dtype=np.object_)
				# mask invalid cells
				classes[ds.col_attrs["_Valid"] == 0] = "Excluded"
				classes_pooled[ds.col_attrs["_Valid"] == 0] = "Excluded"
				classes_pooled[classes_pooled == "None"] = "Excluded"
				ds.set_attr("Class", classes_pooled.astype('str'), axis=1)
				ds.set_attr("Subclass", classes.astype('str'), axis=1)
				for ix, cls in enumerate(class_labels):
					ds.set_attr("ClassProbability_" + str(cls), probs[:, ix], axis=1)
			else:
				logging.info("No classifier found in this build directory - skipping.")
				ds.set_attr("Class", np.array(["Excluded"] * ds.shape[1]), axis=1)
				ds.set_attr("Subclass", np.array(["Unknown"] * ds.shape[1]), axis=1)
			ds.close()
