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
	Luigi Task to run all Level 2 analyses for adolescent mouse
	"""

	def requires(self) -> Iterator[luigi.Task]:
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		classes = ["Oligos", "Astrocyte", "Cycling", "Vascular", "Immune", "Ependymal"]
		for tissue in tissues:
			yield cg.PlotCVMeanL2(project="Adolescent", tissue=tissue, major_class="Neurons")
			yield cg.PlotGraphL2(project="Adolescent", tissue=tissue, major_class="Neurons")
			yield cg.MarkerEnrichmentL2(project="Adolescent", tissue=tissue, major_class="Neurons")
		for cls in classes:
			yield cg.PlotCVMeanL2(project="Adolescent", tissue="All", major_class=cls)
			yield cg.PlotGraphL2(project="Adolescent", tissue="All", major_class=cls)
			yield cg.MarkerEnrichmentL2(project="Adolescent", tissue="All", major_class=cls)
