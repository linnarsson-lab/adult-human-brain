from typing import *
import os
import csv
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi
from collections import defaultdict


class StudyProcessPool(luigi.Task):
	"""
	Luigi Task to generate a particular slice of the data as specified by a description file

	`processname` needs to match th name specified in the .yaml file in the folder ../dev-processes
	"""
	
	processname = luigi.Parameter()

	def requires(self) -> Iterator[luigi.Task]:
		process_obj = cg.ProcessesParser()[self.processname]
		return cg.parse_project_requirements(process_obj)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "%s.loom" % (self.processname,)))
		
	def run(self) -> None:
		process_obj = cg.ProcessesParser()[self.processname]
		logging.debug("Generating the pooled file %s.loom" % self.processname)
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			# The following assumes assumes that for every process taskwrapper the
			# first tow requirements yielded are the clustering and the autoannotation
			# For more flexibility return a dictionary
			for clustered, autoannotated, *others in self.input():
				logging.debug("Adding cells from the source file %s" % clustered.fn)
				ds = loompy.connect(clustered.fn)
				
				# Select the tags as specified in the process file
				filter_bool = cg.FilterManager(process_obj, ds, autoannotated.fn).compute_filter()

				for (ix, selection, vals) in ds.batch_scan_layers(axis=1):
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
						dsout = loompy.create(out_file, m["@DEFAULT"], ds.row_attrs, ca)
						# Add layers
						for layer_name, chunk_of_matrix in m.items():
							if layer_name == "@DEFAULT":
								continue
							dsout.set_layer(layer_name, chunk_of_matrix, dtype=chunk_of_matrix.dtype)
					else:
						dsout.add_columns(m, ca)
