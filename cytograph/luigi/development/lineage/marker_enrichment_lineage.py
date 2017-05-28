from typing import *
import os
import loompy
import numpy as np
import cytograph as cg
import luigi


class MarkerEnrichmentLineage(luigi.Task):
	"""
	Luigi Task to calculate marker enrichment per cluster, level 2
	"""
	lineage = luigi.Parameter(default="Ectodermal")
	target = luigi.Parameter(default="All")
	time = luigi.Parameter(default="E7-E18")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutDev(lineage=self.lineage, target=self.target, time=self.time)
		
	def output(self) -> luigi.Target:
		if self.time == "E7-E18":  # This is for backwards comaptibility we might remove this condition later
			return luigi.LocalTarget(os.path.join(cg.paths().build, self.lineage + "_" + self.target + ".enrichment.tab"))
		else:
			return luigi.LocalTarget(os.path.join(cg.paths().build, "%s_%s_%s.enrichment.tab" % (self.lineage, self.target, self.time)))

	def run(self) -> None:
		with self.output().temporary_path() as f:
			ds = loompy.connect(self.input().fn)
			me = cg.MarkerEnrichment(power=1.0)
			me.fit(ds)
			me.save(f)
