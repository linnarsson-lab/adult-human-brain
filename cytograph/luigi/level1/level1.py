from typing import *
import os
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class Level1(luigi.WrapperTask):
	"""
	Luigi Task to run a subset of level 1 Analysis
	"""

	project = luigi.Parameter(default="Development")
	target = luigi.Parameter(default="All")  # one between Cortex, AllForebrain, ForebrainDorsal, ForebrainVentrolateral, ForebrainVentrothalamic, Midbrain, Hindbrain
	time = luigi.Parameter(default="E7-E18")  # later more specific autoannotation can be devised

	def requires(self) -> Iterator[luigi.Task]:
		tissues = cg.targets_map[self.target]
		for tissue in tissues:
			if cg.time_check(tissue, self.time):
				yield cg.ClusterLayoutL1(tissue=tissue), cg.AutoAnnotateL1(tissue=tissue), cg.PlotGraphL1(tissue=tissue)