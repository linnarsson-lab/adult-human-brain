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


def read_autoannotation(aa_file: str) -> List[List[str]]:
	"""Extract autoannotations from file

	Arguments

	Returns
	-------
	tags : List[List[str]]
		where tags[i] contains all the aa tags attributed to cluster i
	"""
	tags = []  # type: list
	with open(aa_file, "r") as f:
		content = f.readlines()[1:]
		for line in content:
			tags.append(line.rstrip("\n").split('\t')[1].split(","))
	return tags


def make_filter_aa(aa_file_name: str, process_obj: Dict[str, Any]) -> np.ndarray:

	# Read the autoannotation.aa.tab file and extract tags
	tags_per_cluster = read_autoannotation(aa_file_name)
	# Read the process dictionary
	include_aa, exclude_aa = process_obj["include"]["aa"], process_obj["exclude"]["aa"]
	# Add and then remove cluster on the basis of the autoannotation
	selected_clusters = set()  # type: set
	for cluster_ix, tags in enumerate(tags_per_cluster):
		for include_entry in include_aa:
			if type(include_entry) == list:
				if np.alltrue(np.in1d(include_entry, tags)):
					selected_clusters |= {cluster_ix}
			elif type(include_entry) == str:
				if include_entry in tags:
					selected_clusters |= {cluster_ix}
			else:
				logging.warning("Processes: include aa are not correctly fomratted")
		for exclude_entry in exclude_aa:
			if type(exclude_entry) == list:
				if np.alltrue(np.in1d(exclude_entry, tags)):
					selected_clusters -= {cluster_ix}
			elif type(exclude_entry) == str:
				if include_entry in tags:
					selected_clusters -= {cluster_ix}
			else:
				logging.warning("Processes: exclude aa are not correctly fomratted")
	bool_autoannotation = np.in1d(np.arange(len(tags_per_cluster)), list(selected_clusters))
	return bool_autoannotation


class StudyProcessPool(luigi.Task):
	"""
	Luigi Task to generate a particular slice of the data as specified by a description file

	`processname` needs to match th name specified in the .yaml file in the folder ../dev-processes
	"""
	
	processname = luigi.Parameter()

	def requires(self) -> Iterator[luigi.Task]:
		process_obj = cg.ProcessesParser()[self.processname]
		return parse_project_requirements(process_obj)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "%s.loom" % (self.processname,)))
		
	def run(self) -> None:
		# The following code needs to be updated whenever autoannotation is updated
		aa = cg.AutoAnnotator()
		aa.load_defs()
		process_obj = cg.ProcessesParser()[self.processname]
		
		categories_dict = defaultdict(list)  # type: DefaultDict
		for t in aa.tags:
			for c in t.categories:
				categories_dict[c].append(t.abbreviation)

		lineage_abbr = categories_dict[self.lineage]

		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			for clustered, autoannotated, *others in self.input():
				ds = loompy.connect(clustered.fn)
				labels = ds.col_attrs["Clusters"]
				
				# Select the tags as specified in the process file
				bool_autoannotation = make_filter_aa(autoannotated.fn, process_obj)
				bool_category = make_filter_category(ds, process_obj)
				bool_cluster = make_filter_cluster(labels, process_obj)
				bool_classifier = make_filter_classifier()
				filter_bool = bool_autoannotation & bool_category & bool_cluster & bool_classifier

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
