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


class PrepareTimepointPool(luigi.Task):
    """
    Luigi Task to prepare timepointwise files from raw sample files, including gene and cell validation
    """
    timepool = luigi.Parameter()
    tissue = luigi.Parameter(default=None)

    def requires(self) -> List[luigi.Task]:
        if self.tissue:
            samples = cg.PoolSpec().samples_for_tissue_and_timepool(self.tissue, self.timepool)
        else:
            samples = cg.PoolSpec().samples_for_timepool(self.timepool)
        return [cg.Sample(sample=s) for s in samples]

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(os.path.join(cg.paths().build, f"T0_{self.timepool}{'_' + self.tissue if self.tissue else ''}.loom"))

    def run(self) -> None:
        with self.output().temporary_path() as out_file:
            attrs = {"title": f"T0_{self.timepool}{'_' + self.tissue if self.tissue else ''}"}
            valid_cells = []
            sample_files = [s.fn for s in self.input()]
            for sample in sample_files:
                # Connect and perform file-specific cell validation
                logging.info("Marking invalid cells")
                ds = loompy.connect(sample)
                (mols, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
                valid_cells.append(np.logical_and(mols >= 600, (mols / genes) >= 1.2).astype('int'))
                ds.set_attr("_Total", mols, axis=1)
                ds.set_attr("_NGenes", genes, axis=1)
                
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
            
            classifier_loaded = False
            classifier_path = os.path.join(cg.paths().build, "classifier.pickle")
            if os.path.exists(classifier_path):
                try:
                    with open(classifier_path, "rb") as f:
                        clf = pickle.load(f)  # type: cg.Classifier
                    classes = clf.predict(ds)
                    classifier_loaded = True
                except (pickle.UnpicklingError, UnicodeDecodeError) as e:
                    logging.error("Error during Clasifier Loading! Continuing without.")
                except ValueError as e:
                    logging.error("Error during Clasifier classification! Message:%s" % e)

            if classifier_loaded:
                logging.info("Classifying cells by major class")
                
                mapping = {
                    "Astrocyte": "Astrocytes",
                    "Astrocyte,Cycling": "Astrocytes",
                    "Astrocyte,Immune": None,
                    "Astrocyte,Neurons": None,
                    "Astrocyte,Oligos": None,
                    "Astrocyte,Vascular": None,
                    "Bergmann-glia": "Astrocytes",
                    "Blood": "Blood",
                    "Blood,Cycling": "Blood",
                    "Blood,Vascular": None,
                    "Ependymal": "Ependymal",
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
                    "OEC": "Astrocytes",
                    "Oligos": "Oligos",
                    "Oligos,Cycling": "Oligos",
                    "Oligos,Immune": None,
                    "Oligos,Neurons": None,
                    "Oligos,Vascular": None,
                    "Satellite-glia": "PeripheralGlia",
                    "Satellite-glia,Cycling": "PeripheralGlia",
                    "Schwann": "PeripheralGlia",
                    "Schwann,Satellite-glia": None,
                    "Ttr": "Ependymal",
                    "Vascular": "Vascular",
                    "Vascular,Cycling": "Vascular",
                    "Vascular,Neurons": None,
                    "Vascular,Oligos": None,
                    "Vascular,Satellite-glia": None,
                    "Unknown": None
                }

                classes_pooled = np.array([str(mapping[c]) for c in classes], dtype=np.object_)
                # mask invalid cells
                classes[ds.col_attrs["_Valid"] == 0] = "Excluded"
                classes_pooled[ds.col_attrs["_Valid"] == 0] = "Excluded"
                classes_pooled[classes_pooled == "None"] = "Excluded"
                ds.set_attr("Class", classes_pooled.astype('str'), axis=1)
                ds.set_attr("Subclass", classes.astype('str'), axis=1)
            else:
                logging.info("No classifier found in this build directory - skipping.")
                ds.set_attr("Class", np.array(["Excluded"] * ds.shape[1]), axis=1)
                ds.set_attr("Subclass", np.array(["Unknown"] * ds.shape[1]), axis=1)
            ds.close()
