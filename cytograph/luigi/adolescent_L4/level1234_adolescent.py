from typing import *
import os
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi


class Level1234Adolescent(luigi.WrapperTask):
	"""
	Luigi Task to run all Level 1 and 2 analyses
	"""

	def requires(self) -> Iterator[luigi.Task]:
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		classes = ["Oligos", "Ependymal", "Astrocytes", "Vascular", "Immune", "PeripheralGlia"]
		for tissue in tissues:
			yield cg.ExportL1(tissue=tissue)
			yield cg.ExportL2(tissue=tissue, major_class="Neurons")
			yield cg.ExportL3(tissue=tissue, major_class="Neurons")

		for cls in classes:
			yield cg.ExportL2(tissue="All", major_class=cls)
			yield cg.ExportL3(tissue="All", major_class=cls)
		yield cg.ExportL4()
