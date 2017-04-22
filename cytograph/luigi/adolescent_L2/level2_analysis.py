from typing import *
import os
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class Level2Analysis(luigi.WrapperTask):
	"""
	Luigi Task to run all Level 2 analyses
	"""
	project = luigi.Parameter(default="Adolescent")

	def requires(self) -> Iterator[luigi.Task]:
		if self.project == "Adolescent":
			tissues = cg.PoolSpec().tissues_for_project(self.project)
			for tissue in tissues:
				yield cg.PlotCVMeanL2(project=self.project, tissue=tissue, major_class="Neurons")
				yield cg.PlotGraphL2(project=self.project, tissue=tissue, major_class="Neurons")
				yield cg.MarkerEnrichmentL2(project=self.project, tissue=tissue, major_class="Neurons")
				yield cg.PlotClassesL2(project=self.project, tissue=tissue, major_class="Neurons")
				yield cg.PlotMarkerheatmapL2(project=self.project, tissue=tissue, major_class="Neurons")

			# for tissue in tissues:
			# 	yield cg.PlotCVMeanL2(project=self.project, tissue=tissue, major_class="Excluded")
			# 	yield cg.PlotGraphL2(project=self.project, tissue=tissue, major_class="Excluded")
			# 	yield cg.MarkerEnrichmentL2(project=self.project, tissue=tissue, major_class="Excluded")
			# 	yield cg.PlotClassesL2(project=self.project, tissue=tissue, major_class="Excluded")

			classes = ["Oligos", "Astrocyte", "Cycling", "Vascular", "Immune", "Erythrocyte"]  # "Excluded"]
			for cls in classes:
				yield cg.PlotCVMeanL2(project=self.project, tissue="All", major_class=cls)
				yield cg.PlotGraphL2(project=self.project, tissue="All", major_class=cls)
				yield cg.MarkerEnrichmentL2(project=self.project, tissue="All", major_class=cls)
				yield cg.PlotClassesL2(project=self.project, tissue="All", major_class=cls)
				yield cg.PlotMarkerheatmapL2(project=self.project, tissue="All", major_class=cls)
		else:
			yield cg.PlotCVMeanL2(project=self.project, tissue="All", major_class="Development")
			yield cg.PlotGraphL2(project=self.project, tissue="All", major_class="Development")
			yield cg.MarkerEnrichmentL2(project=self.project, tissue="All", major_class="Development")
			yield cg.PlotClassesL2(project=self.project, tissue="All", major_class="Development")
			yield cg.PlotMarkerheatmapL2(project=self.project, tissue="All", major_class="Development")

