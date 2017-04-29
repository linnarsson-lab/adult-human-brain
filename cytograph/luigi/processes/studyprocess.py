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
from luigi import targets_map, EP2int, time_check, analysis_type_dict
from collections import defaultdict


def parse_project_requirements(process_obj: Dict) -> Iterator[luigi.Task]:
	parent_type = process_obj["type"]
	parent_kwargs = process_obj["kwargs"]
	if parent_type not in analysis_type_dict:
		raise NotImplementedError("type: %s not allowed, you need to allow it adding it to analysis_type_dict" % parent_type)
	Analysis = analysis_type_dict[parent_type]
	return Analysis(**parent_kwargs).requires()


class StudyProcess(luigi.Task):
	"""
	Luigi Task Wrapper to run a set of analysese on a particular slice of the data as specified by a description file

	`processname` needs to match th name specified in the .yaml file in the folder ../dev-processes
	"""
	
	processname = luigi.Parameter()

	def requires(self) -> Iterator[luigi.Task]:
		process_obj = cg.ProcessesParser()[self.processname]
		return parse_project_requirements(process_obj)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "%s_%s_%s.loom" % (self.lineage, self.target, self.time)))
		
	def run(self) -> None:
		# The following code needs to be updated whenever autoannotation is updated
		aa = cg.AutoAnnotator()
		aa.load_defs()
		categories_dict = defaultdict(list)  # type: DefaultDict
		for t in aa.tags:
			for c in t.categories:
				categories_dict[c].append(t.abbreviation)

		lineage_abbr = categories_dict[self.lineage]

		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			for clustered, autoannotated in self.input():
				ds = loompy.connect(clustered.fn)
				labels = ds.col_attrs["Clusters"]
				tags = []
				# Read the autoannotation.aa.tab file and extract tags
				with open(autoannotated.fn, "r") as f:
					content = f.readlines()[1:]
					for line in content:
						tags.append(line.rstrip("\n").split('\t')[1].split(","))
				# Select the tags that belong to the lineage
				selected_tags = []
				for i, t in enumerate(tags):
					if np.any(np.in1d(t, lineage_abbr)):
						selected_tags.append(i)
				for (ix, selection, vals) in ds.batch_scan(axis=1):
					# Filter the cells that belong to the selected tags
					subset = np.intersect1d(np.where(np.in1d(labels, selected_tags))[0], selection)
					if subset.shape[0] == 0:
						continue
					m = vals[:, subset - ix]
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][subset]
					# Add data to the loom file
					if dsout is None:
						dsout = loompy.create(out_file, m, ds.row_attrs, ca)
					else:
						dsout.add_columns(m, ca)
