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


class SplitAndPoolAa(luigi.Task):
	"""
	Luigi Task to split the results of level 1 analysis by using both the region and auto-annotion, and pool each class separately

	`lineage` cane be:
	Ectodermal (default), Endomesodermal

	`target` can be:
	All (default), Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral, ForebrainVentrothalamic, Midbrain, Hindbrain
	"""
	# project = luigi.Parameter(default="Development")  # For now this works only for development
	lineage = luigi.Parameter(default="Ectodermal")  # Alternativelly Endomesodermal
	target = luigi.Parameter(default="All")  # one between Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral, ForebrainVentrothalamic, Midbrain, Hindbrain

	def requires(self) -> luigi.Task:
		targets_map = {
			"All": [
				'Cephalic_E7-8', 'Forebrain_E9-11', 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18',
				'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18', 'ForebrainVentrothalamic_E16-18',
				'Midbrain_E9-11', 'Midbrain_E12-15', 'Midbrain_E16-18', 'Hindbrain_E9-11', 'Hindbrain_E12-15',
				'Hindbrain_E16-18'],
			"AllForebrain": [
				"Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18',
				'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18', 'ForebrainVentrothalamic_E16-18'],
			"ForebrainDorsal": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18'],
			"ForebrainVentrolateral": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainVentral_E12-15', 'ForebrainVentrolateral_E16-18'],
			"ForebrainVentrothalamic": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainVentral_E12-15', 'ForebrainVentrothalamic_E16-18'],
			"Midbrain": ["Cephalic_E7-8", 'Midbrain_E9-11', 'Midbrain_E12-15', 'Midbrain_E16-18'],
			"Hindbrain": ["Cephalic_E7-8", 'Hindbrain_E9-11', 'Hindbrain_E12-15', 'Hindbrain_E16-18'],
			"Cortex": ["Cephalic_E7-8", "Forebrain_E9-11", 'ForebrainDorsal_E12-15', 'ForebrainDorsal_E16-18', "Cephalic_E7-8"]}
		return [[cg.ClusterLayoutL1(tissue=tissue), cg.AutoAnnotateL1(tissue=tissue)] for tissue in targets_map[self.target]]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.lineage + "_" + self.target + ".loom"))
		
	def run(self) -> None:
		# The following code needs to be updated whenever autoannotation is updated
		ectodermal_abbr = [
			'NProgF', 'ANPlt', 'SCHWA', 'DG-MSY', 'OB-OEC', '@OL', 'AC', '@IEG', 'CRC', '@p2AL', '@SER', 'IEar',
			'HRgl3', '@DA', 'CNC', 'CrMN', 'NblastL1', '@Pall', 'PSN', '@VGLUT2', 'PyrL6b', 'NblastM', 'Nose', '@pTh',
			'NProg', '@N', '@GABA', '@DMid', 'EPN', 'DG-GC', '@CHind', 'COP', '@Eye', '@Habe', 'CHRD', 'EHProg', '@BMono',
			'@PMid', 'MGL', 'Rgl', 'OPC', 'FGut', 'AEMeso', 'PyrL6b', 'PyrL4-5a', '@RDie', 'FrontM', 'OLIG', 'StemM']
		endomesodermal_abbr = [
			'PERI', '@Msn', 'Dermo', 'ExVE', 'AVE', 'CFacM', 'PxMeso', 'DVE', 'CrPr', 'PLAT', 'VSM', 'LEUKO', 'GUM',
			'Tropho', 'MFIB', 'VLMC', '@Angio', 'EDEndo', 'CARD', 'Angio', 'VEC', 'VEC-FC', 'CrCr', 'Eryp', 'EFace',
			'LPMeso', 'PVM', 'ERY', 'OstCho']
		lineage_abbr = {"Ectodermal": ectodermal_abbr, "Endomesodermal": endomesodermal_abbr}[self.lineage]

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
					if np.any(np.in1d(i, lineage_abbr)):
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
