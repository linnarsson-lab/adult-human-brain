from typing import *
import os
import loompy
import numpy as np
import cytograph as cg
import luigi


class MarkerEnrichmentL3(luigi.Task):
	"""
	Luigi Task to calculate marker enrichment per cluster, level 3
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutL2(tissue=self.tissue, major_class=self.major_class, project=self.project)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".enrichment.tab"))

	def run(self) -> None:
		with self.output().temporary_path() as f:
			ds = loompy.connect(self.input().fn)
			me = cg.MarkerEnrichment(power=1.0)
			me.fit(ds)
			me.save(f)
