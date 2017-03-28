from typing import *
import os
import csv
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi
import scipy.cluster.hierarchy as hierarchy


class AggregateClusters(luigi.Task):
	"""
	Aggregate all clusters in a new Loom file
	"""
	project = luigi.Parameter(default="Adolescent")

	def requires(self) -> Iterator[luigi.Task]:
		if self.project == "Adolescent":
			tissues = cg.PoolSpec().tissues_for_project(self.project)
			classes = ["Oligos", "Astrocyte", "Cycling", "Vascular", "Immune", "Erythrocyte"]
			for tissue in tissues:
				yield cg.ClusterLayoutL2(project=self.project, tissue=tissue, major_class="Neurons")
			for cls in classes:
				yield cg.ClusterLayoutL2(project=self.project, tissue="All", major_class=cls)
		else:
			yield cg.ClusterLayoutL2(project=self.project, tissue="All", major_class="Development")

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.project + ".aggregated.L3.loom"))
		
	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			for f in self.input():
				ds = loompy.connect(f.fn)
				cg.aggregate_loom(ds, out_file, )