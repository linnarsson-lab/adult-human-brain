from typing import *
import os
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi


class Level123Adolescent(luigi.WrapperTask):
	"""
	Luigi Task to run all Level 1 - 3 analyses
	"""

	def requires(self) -> Iterator[luigi.Task]:
		targets = [
			'Peripheral_Neurons',
			'DiMesencephalon_Excitatory',
			'Hindbrain_Inhibitory',
			'SpinalCord_Inhibitory',
			'Brain_Granule',
			'Brain_CholinergicMonoaminergic',
			'DiMesencephalon_Inhibitory',
			'Striatum_MSN',
			'Hypothalamus_Peptidergic',
			'Forebrain_Excitatory',
			'Forebrain_Neuroblasts',
			'Hindbrain_Excitatory',
			'SpinalCord_Excitatory',
			'Forebrain_Inhibitory'
		]
		tissues = cg.PoolSpec().tissues_for_project("Adolescent")
		classes = ["Oligos", "AstroEpendymal", "Vascular", "Immune", "Blood", "PeripheralGlia"]
		for tissue in tissues:
			yield cg.ExportL1(tissue=tissue)
			yield cg.ExportL2(tissue=tissue, major_class="Neurons")

		for cls in classes:
			yield cg.ExportL2(tissue="All", major_class=cls)

		for target in targets:
			yield cg.ExportL3(target=target)
