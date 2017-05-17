from typing import *
import os
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class Level2Adolescent(luigi.WrapperTask):
	"""
	Luigi Task to run all Level 2 analyses
	"""

	def requires(self) -> Iterator[luigi.Task]:
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		for tissue in tissues:
			yield cg.PlotManifoldL2(tissue=tissue, major_class="Neurons")
			yield cg.PlotMarkerheatmapL2(tissue=tissue, major_class="Neurons")

		# for tissue in tissues:
		# 	yield cg.PlotCVMeanL2(tissue=tissue, major_class="Excluded")
		# 	yield cg.PlotGraphL2(tissue=tissue, major_class="Excluded")
		# 	yield cg.MarkerEnrichmentL2(tissue=tissue, major_class="Excluded")
		# 	yield cg.PlotClassesL2(tissue=tissue, major_class="Excluded")

		classes = ["Oligos", "Astrocyte", "Cycling", "Vascular", "Immune", "Erythrocyte"]  # "Excluded"]
		for cls in classes:
			yield cg.PlotManifoldL2(tissue="All", major_class=cls)
			yield cg.PlotMarkerheatmapL2(tissue="All", major_class=cls)
