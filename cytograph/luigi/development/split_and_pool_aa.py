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


class SplitAndPoolAa(luigi.Task):
	"""
	Luigi Task to split the results of level 1 analysis by using both the region, the auto-annotion and the time, and pool each class separately

	`lineage` cane be:
	Ectodermal (default), Endomesodermal

	`target` can be:
	All (default), Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral, ForebrainVentrothalamic, Midbrain, Hindbrain

	`time` can be:
	EarlyTime-LaterTime, for example E9-E18, (EarlyTime allowed are E7, E9, E12, E16; LaterTime allowed are E8, E11, E15, E18, P7; with EarlyTime < LaterTime)
	default: E7-E18
	"""
	# project = luigi.Parameter(default="Development")  # For now this works only for development
	lineage = luigi.Parameter(default="Ectodermal")  # One of the categories in the autoannotation files
	target = luigi.Parameter(default="All")  # one between Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral, ForebrainVentrothalamic, Midbrain, Hindbrain
	time = luigi.Parameter(default="E7-E18")  # later more specific autoannotation can be devised

	def requires(self) -> luigi.Task:
		return [[cg.ClusterLayoutL1(tissue=tissue), cg.AutoAnnotateL1(tissue=tissue)] for tissue in cg.targets_map[self.target] if cg.time_check(tissue, self.time)]

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".loom"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.loom" % (self.lineage, self.target, self.time)))
		
	def run(self) -> None:
		# The following code needs to be updated whenever autoannotation is updated
		aa = cg.AutoAnnotator.load_direct()
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
				for (ix, selection, vals) in ds.batch_scan(axis=1, batch_size=cg.memory().axis1):
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
