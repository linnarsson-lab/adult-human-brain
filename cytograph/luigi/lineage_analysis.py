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
	"""
	lineage = luigi.Parameter(default="Ectodermal")  # `All` or one of the allowed lineage parameters of SplitAndPoolAa (currently Ectodermal, Endomesodermal)
	targetset = luigi.Parameter(default="MainRegions")  # MainRegions, AllMerged, ForebrainMerged, Postnatal, Everything

	def requires(self) -> Iterator[luigi.Task]:
		if self.lineage == "All":
			lineages = ["Ectodermal", "Endomesodermal"]
		else:
			lineages = [self.lineage]

		if self.targetset == "MainRegions":
			targets = ["ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain"]
		elif self.targetset == "AllMerged":
			targets = ["All"]
		elif self.targetset == "ForebrainMerged":
			targets = ["AllForebrain"]
		elif self.targetset == "Postnatal":
			targets = ["Cortex"]
		elif self.targetset == "Everything":
			targets = [
				"ForebrainDorsal", "ForebrainVentrolateral", "ForebrainVentrothalamic", "Midbrain", "Hindbrain",
				"All", "AllForebrain", "Cortex"]

		for ll in lineages:
			for tt in targets:
				yield cg.PlotCVMeanLineage(lineage=ll, target=tt)
				yield cg.PlotGraphLineage(lineage=ll, target=tt)
				yield cg.MarkerEnrichmentLineage(lineage=ll, target=tt)
				yield cg.PlotClassesLineage(lineage=ll, target=tt)
