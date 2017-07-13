from typing import *
import os
import loompy
import numpy as np
import cytograph as cg
import luigi


class MarkerEnrichmentProcess(luigi.Task):
	"""
	Luigi Task to calculate marker enrichment per cluster, level 2
	"""
	processname = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutProcess(processname=self.processname)
		
	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "%s.enrichment.tab" % self.processname))

	def run(self) -> None:
		with self.output().temporary_path() as f:
			ds = loompy.connect(self.input().fn)
			me = cg.MarkerEnrichment(power=1.0)
			me.fit(ds)
			me.save(f)
