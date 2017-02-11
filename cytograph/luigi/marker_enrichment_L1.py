from typing import *
import os
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class MarkerEnrichmentL1(luigi.Task):
	"""
	Luigi Task to calculate marker enrichment per cluster
	"""
	tissue = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutL1(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.tissue + ".enrichment.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as f:
			ds = loompy.connect(self.input().fn)
			me = cg.MarkerEnrichment(power=1.0)
			me.fit(ds)
			me.save(f)
