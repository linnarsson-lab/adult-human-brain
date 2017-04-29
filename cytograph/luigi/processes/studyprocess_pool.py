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


class FilterManager(object):
	def __init__(self, process_obj: Dict, ds: loompy.LoomConnection, aa_file_name: str=None) -> None:
		self.process_obj = process_obj
		self.ds = ds
		self.aa_file_name = aa_file_name
	
	def make_filter_aa(self) -> Tuple[np.ndarray, np.ndarray]:
		# Read the autoannotation.aa.tab file and extract tags
		tags_per_cluster = cg.read_autoannotation(self.aa_file_name)
		# Read the process dictionary
		include_aa = self.process_obj["include"]["auto-annotations"]
		exclude_aa = self.process_obj["exclude"]["auto-annotations"]
		# Add and then remove cluster on the basis of the autoannotation
		selected_clusters = set()  # type: set
		deselected_clusters = set()  # type: set
		for cluster_ix, tags in enumerate(tags_per_cluster):
			# Deal with the inclusions
			if include_aa == "all":
				selected_clusters = set(list(range(len(tags_per_cluster))))
			else:
				for include_entry in include_aa:
					if type(include_entry) == list:
						if np.alltrue(np.in1d(include_entry, tags)):
							selected_clusters |= {cluster_ix}
					elif type(include_entry) == str:
						if include_entry in tags:
							selected_clusters |= {cluster_ix}
					else:
						logging.warning("Processes: include aa are not correctly fomratted")
			# Deal with the exclusions
			if exclude_aa == "none":
				deselected_clusters = set()
			else:
				for exclude_entry in exclude_aa:
					if type(exclude_entry) == list:
						if np.alltrue(np.in1d(exclude_entry, tags)):
							deselected_clusters |= {cluster_ix}
					elif type(exclude_entry) == str:
						if include_entry in tags:
							deselected_clusters |= {cluster_ix}
					else:
						logging.warning("Processes: exclude aa are not correctly fomratted")
		in_aa = np.in1d(self.ds.col_attrs["Clusters"], list(selected_clusters))
		ex_aa = np.in1d(self.ds.col_attrs["Clusters"], list(deselected_clusters))
		return in_aa, ex_aa

	def make_filter_classifier(self) -> Tuple[np.ndarray, np.ndarray]:
		include_class = self.process_obj["include"]["classes"]
		exclude_class = self.process_obj["exclude"]["classes"]
		in_cla = np.zeros(self.ds.shape[0], dtype=bool)
		ex_cla = np.zeros(self.ds.shape[0], dtype=bool)
		# Deals with inclusions
		if include_class == "all":
			in_cla = np.ones(self.ds.shape[0], dtype=bool)
		else:
			for cl in include_class:
				in_cla |= self.ds.col_attrs["Class_%s" % cl.title()] > 0.5
		# Deals with exclusions
		if exclude_class == "none":
			pass
		else:
			for cl in exclude_class:
				ex_cla |= self.ds.col_attrs["Class_%s" % cl.title()] > 0.5

		return in_cla, ex_cla

	def make_filter_cluster(self) -> Tuple[np.ndarray, np.ndarray]:
		include_clust = self.process_obj["include"]["clusters"]
		exclude_clust = self.process_obj["exclude"]["clusters"]
		# Deals with inclusions
		if include_clust == "all":
			in_clu = np.ones(self.ds.shape[0], dtype=bool)
		else:
			in_clu = np.in1d(self.ds.col_attrs["Clusters"], include_clust)
		# Deals with exclusions
		if exclude_clust == "none":
			ex_clu = np.zeros(self.ds.shape[0], dtype=bool)
		else:
			ex_clu = np.in1d(self.ds.col_attrs["Clusters"], exclude_clust)

		return in_clu, ex_clu

	def make_filter_category(self) -> np.ndarray:
		aa = cg.AutoAnnotator()
		aa.load_defs()
		categories_dict = defaultdict(list)  # type: DefaultDict
		for t in aa.tags:
			for c in t.categories:
				categories_dict[c].append(t.abbreviation)
		include_cat = self.process_obj["include"]["categories"]
		exclude_cat = self.process_obj["exclude"]["categories"]

		include_aa = []  # type: list
		for cat in include_cat:
			if type(cat) == str:
				include_aa += categories_dict[cat]
			elif type(cat) == list:
				intersection = set(categories_dict[cat[0]])
				for c in cat[1:]:
					intersection &= set(categories_dict[c])
				include_aa += list(intersection)
			else:
				logging.warning("Processes: exclude categories are not correctly formatted")
		
		exclude_aa = self.process_obj["exclude"]["auto-annotations"]

	def compute_filter(self) -> np.ndarray:
		in_aa, ex_aa = self.make_filter_aa()
		in_cat, ex_cat = self.make_filter_category()
		in_clu, ex_clu = self.make_filter_cluster()
		in_cla, ex_cla = self.make_filter_classifier()
		filter_include = (in_aa | in_cat | in_clu | in_cla)
		filter_exclude = (ex_aa | ex_cat | ex_clu | ex_cla)
		return filter_include & np.logical_not(filter_exclude)


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
		process_obj = cg.ProcessesParser()[self.processname]

		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			for clustered, autoannotated, *others in self.input():
				ds = loompy.connect(clustered.fn)
				labels = ds.col_attrs["Clusters"]
				
				# Select the tags as specified in the process file
				filter_bool = FilterManager(process_obj, ds, autoannotated.fn).compute_filter()

				for (ix, selection, vals) in ds.batch_scan(axis=1):
					# Filter the cells that belong to the selected tags
					subset = np.intersect1d(np.where(filter_bool)[0], selection)
					if subset.shape[0] == 0:
						continue
					m = vals[:, subset - ix]
					ca = {}
					for key in ds.col_attrs:
						ca[key] = ds.col_attrs[key][subset]
					# Add data to the loom file
					if dsout is None:
						dsout = loompy.create(out_file, m, ds.row_attrs, ca)
						# Add layer!
					else:
						dsout.add_columns(m, ca)
