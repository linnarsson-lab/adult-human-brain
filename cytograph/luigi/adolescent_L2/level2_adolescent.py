from typing import *
import os
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi


class Level2Adolescent(luigi.WrapperTask):
	"""
	Luigi Task to run all Level 2 analyses
	"""

	def requires(self) -> Iterator[luigi.Task]:
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		classes = ["Neurons", "Oligos", "AstroEpendymal", "Vascular", "Immune", "Blood", "PeripheralGlia"]  # "Excluded"]
		for tissue in tissues:
			for cls in classes:
				yield cg.PlotManifoldL2(tissue=tissue, major_class=cls)
				yield cg.PlotMarkerheatmapL2(tissue=tissue, major_class=cls)

		# classes = ["Oligos", "Astrocyte", "Cycling", "Vascular", "Immune", "Erythrocyte"]  # "Excluded"]
		# for cls in classes:
		# 	yield cg.PlotManifoldL2(tissue="All", major_class=cls)
		# 	yield cg.PlotMarkerheatmapL2(tissue="All", major_class=cls)
