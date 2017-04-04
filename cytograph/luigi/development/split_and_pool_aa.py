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

		def time_check(tissue_name: str, time_par: str) -> bool:
			earlytime, latertime = time_par.split("-")
			try:
				tissue_earlytime, tissue_latertime = tissue_name.split("_")[-1].split("-")
			except ValueError:
				tissue_earlytime = tissue_name.split("_")[-1]
				tissue_latertime = tissue_earlytime
			return (earlytime <= tissue_earlytime) and (latertime >= tissue_latertime)

		targets_map = {
			"All": [
				'Cephalic_E7-8', 'Forebrain_E9-11', 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18',
				'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18', 'ForebrainVentrothalamic_E16-18',
				'Midbrain_E9-11', 'Midbrain_E12-15', 'Midbrain_E16-18', 'Hindbrain_E9-11', 'Hindbrain_E12-15',
				'Hindbrain_E16-18'],
			"ForebrainAll": [
				"Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18',
				'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18', 'ForebrainVentrothalamic_E16-18'],
			"ForebrainDorsal": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18'],
			"ForebrainVentrolateral": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18'],
			"ForebrainVentrothalamic": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainVentral_E12-15', 'ForebrainVentrothalamic_E16-18'],
			"Midbrain": ["Cephalic_E7-8", 'Midbrain_E9-11', 'Midbrain_E12-15', 'Midbrain_E16-18'],
			"Hindbrain": ["Cephalic_E7-8", 'Hindbrain_E9-11', 'Hindbrain_E12-15', 'Hindbrain_E16-18'],
			"Cortex": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18', "Cortex_P7"]}
		return [[cg.ClusterLayoutL1(tissue=tissue), cg.AutoAnnotateL1(tissue=tissue)] for tissue in targets_map[self.target] if time_check(tissue, self.time)]

	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join("loom_builds", self.lineage + "_" + self.target + ".loom"))
		else:
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
