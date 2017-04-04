from typing import *
import os
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class LineageAnalysis(luigi.WrapperTask):
	"""
	Luigi Task to run all Lineage Analysis analyses

	`lineage` cane be:
	Ectodermal (default), Endomesodermal, Radialglialike, Neuroectodermal, Neuronal

	`targetset` can be:
	FMH will run targets: "ForebrainAll", "Midbrain", "Hindbrain"
	MainRegions will run targets: "ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain"
	Postnatal will run targets: "Cortex"
	Everything will run targets: "ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain","All", "ForebrainAll", "Cortex"
	or any of the specific a single target argument:
	"ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain", "All", "ForebrainAll", "Cortex"

	"""
	lineage = luigi.Parameter(default="Ectodermal")  # `All` or one of the allowed lineage parameters of SplitAndPoolAa (currently Ectodermal, Endomesodermal)
	targetset = luigi.Parameter(default="MainRegions")  # MainRegions, AllMerged, ForebrainMerged, Postnatal, Everything

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
				yield cg.PlotCVMeanLineage(lineage=ll, target=tt)
				yield cg.PlotGraphLineage(lineage=ll, target=tt)
				yield cg.PlotGraphAgeLineage(lineage=ll, target=tt)
				yield cg.MarkerEnrichmentLineage(lineage=ll, target=tt)
				yield cg.PlotClassesLineage(lineage=ll, target=tt)
				yield cg.ExpressionAverageLineage(lineage=ll, target=tt)
