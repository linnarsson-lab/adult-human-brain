from typing import *
import os
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi


class TimeAnalysis(luigi.WrapperTask):
	"""
	Luigi Task to run all Time Analysis analyses

	`lineage` cane be:
	Ectodermal (default), Endomesodermal, Radialglialike, Neuroectodermal, Neuronal

	`targetset` can be:
	FMH will run targets: "ForebrainAll", "Midbrain", "Hindbrain"
	MainRegions will run targets: "ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain"
	Postnatal will run targets: "Cortex"
	Everything will run targets: "ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain","All", "ForebrainAll", "Cortex"
	or any of the specific a single target argument:
	"ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain", "All", "ForebrainAll", "Cortex"

	`time` can be:
	EarlyTime-LaterTime, for example E9-E18, (EarlyTime allowed are E7, E9, E12, E16; LaterTime allowed are E8, E11, E15, E18, P7; with EarlyTime < LaterTime)
	default: E7-E18
	"""
	lineage = luigi.Parameter(default="Ectodermal")  # `All` or one of the allowed lineage parameters of SplitAndPoolAa (currently Ectodermal, Endomesodermal)
	targetset = luigi.Parameter(default="FMH")  # MainRegions, AllMerged, ForebrainMerged, Postnatal, Everything
	time = luigi.Parameter(default="E7-E18")

	def requires(self) -> Iterator[luigi.Task]:
		if self.lineage == "All":
			lineages = ["Ectodermal", "Endomesodermal", "Radialglialike", "Neuroectodermal", "Neuronal"]
		else:
			lineages = [self.lineage]
		
		if self.targetset == "FMH":
			targets = ["ForebrainAll", "Midbrain", "Hindbrain"]
		elif self.targetset == "MainRegions":
			targets = ["ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain"]
		elif self.targetset == "Postnatal":
			targets = ["Cortex"]
		elif self.targetset == "Everything":
			targets = ["ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain","All", "ForebrainAll", "Cortex"]
		else:
			if self.targetset in ["ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain", "All", "AllForebrain", "Cortex"]:
				targets = [self.targetset]
			else:
				raise KeyError

		for ll in lineages:
			for tt in targets:
				yield cg.ExpressionAverageTime(lineage=ll, target=tt, time=self.time)
