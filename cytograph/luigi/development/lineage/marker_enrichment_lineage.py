from typing import *
import os
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class MarkerEnrichmentLineage(luigi.Task):
	"""
	Luigi Task to calculate marker enrichment per cluster, level 2
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutDev(lineage=self.lineage, target=self.target)
		
	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.lineage + "_" + self.target + ".enrichment.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as f:
			ds = loompy.connect(self.input().fn)
			me = cg.MarkerEnrichment(power=1.0)
			me.fit(ds)
			me.save(f)
