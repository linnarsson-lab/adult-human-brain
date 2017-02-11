"""
From Amit:


few more things,
before sending the data to the clustering i usually filter out cells like that:

validcells = (tot_mol>600 & (tot_mol./tot_genes) > 1.2 & tot_mol < 20000 & tot_genes > 500);
then i remove doublets based on markers for neurons, oligo,endo,microglia, astro ect.
less relevant maybe to development but can be good to check it.

for genes, first filter
in = find(sum(data>0,2)>20 & sum(data>0,2)<length(data(1,:))*0.6);

then using cv vs mean select 10,000 genes (maybe should be higher)
remove sex genes
shuffle the columns
normalize the total number of molecules to 10,000. i found that this give better results.

amit
"""


import os
from shutil import copyfile
from tempfile import mktemp
from typing import *
import logging
import numpy as np
import loompy


def validate_cells(ds: loompy.LoomConnection) -> None:
	(mols, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
	valid = np.logical_and(np.logical_and(mols >= 600, (mols / genes) >= 1.2), np.logical_and(mols <= 20000, genes >= 500)).astype('int')
	ds.set_attr("_Valid", valid, axis=1)


def validate_genes(ds: loompy.LoomConnection) -> None:
	nnz = ds.map(np.count_nonzero, axis=0)
	valid = np.logical_and(nnz > 20, nnz < ds.shape[1] * 0.6)
	ds.set_attr("_Valid", valid, axis=0)


def preprocess(loom_folder: str, build_folder: str, sample_ids: np.ndarray, out_file: str, attrs: Dict = None, make_doublets: bool = False, do_validate_genes: bool = False) -> Tuple[int, int]:
	if attrs is None:
		attrs = {}

	# Keep track of temporary copies of the loom files
	temp_files = []
	n_valid = 0
	n_total = 0

	for sample_id in sample_ids:
		logging.info("Creating temp file for " + sample_id)

		# Make a temporary loom file name, track it, and copy the sample
		fname = mktemp(suffix=".loom", dir=build_folder)
		temp_files.append(fname)
		copyfile(os.path.join(loom_folder, sample_id + ".loom"), fname)
		logging.info("Preprocessing " + sample_id)

		# Connect and perform file-specific QC and validation
		ds = loompy.connect(fname)

		logging.info("Marking invalid cells")
		validate_cells(ds)
		n_valid += np.sum(ds.col_attrs["_Valid"] == 1)
		n_total += ds.shape[1]

		ds.close()

	logging.info("Creating combined loom file")
	loompy.combine(temp_files, out_file, key="Accession", file_attrs=attrs)
	ds = loompy.connect(out_file)
	if do_validate_genes:
		logging.info("Marking invalid genes")
		validate_genes(ds)
	logging.info("Computing aggregate statistics")
	with np.errstate(divide='ignore', invalid='ignore'):
		ds.compute_stats()
	ds.close()

	# Remove the temporary loom files
	for fname in temp_files:
		os.remove(fname)

	return (n_valid, n_total)
