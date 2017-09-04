from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
from collections import defaultdict


class AnalysisPool(luigi.Task):  # Status: check the filter manager
	"""
	Luigi Task to generate a particular slice of the data as specified by a description file

	`analysis` needs to match the name specified in the .yaml file in the folder ../cg-analysis
	"""
	
	analysis = luigi.Parameter()

	def requires(self) -> Iterator[luigi.Task]:
		analysis_obj = cg.AnalysesParser()[self.analysis]
		return cg.parse_analysis_requirements(analysis_obj)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "%s.loom" % (self.analysis,)))
		
	def run(self) -> None:
		analysis_obj = cg.AnalysesParser()[self.analysis]
		logging.debug("Generating the pooled file %s.loom" % self.analysis)
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			# The following assumes assumes that for every process taskwrapper the
			# first tow requirements yielded are the clustering and the autoannotation
			# For more flexibility return a dictionary
			for clustered, autoannotated, *others in self.input():
				logging.debug("Adding cells from the source file %s" % clustered.fn)
				ds = loompy.connect(clustered.fn)
				
				# Select the tags as specified in the process file
				filter_bool = cg.FilterManager(analysis_obj, ds, autoannotated.fn).compute_filter()

				for (ix, selection, vals) in ds.batch_scan_layers(axis=1, batch_size=cg.memory().axis1):
					# Filter the cells that belong to the selected tags
					subset = np.intersect1d(np.where(filter_bool)[0], selection)
					if subset.shape[0] == 0:
						continue
					m = {}
					for layer_name, chunk_of_matrix in vals.items():
						m[layer_name] = vals[layer_name][:, subset - ix]
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][subset]
					# Add data to the loom file
					if dsout is None:
						# create using main layer
						dsout = loompy.create(out_file, m[""], ds.row_attrs, ca)
						# Add layers
						for layer_name, chunk_of_matrix in m.items():
							if layer_name == "":
								continue
							dsout.set_layer(layer_name, chunk_of_matrix, dtype=chunk_of_matrix.dtype)
					else:
						dsout.add_columns(m, ca)
