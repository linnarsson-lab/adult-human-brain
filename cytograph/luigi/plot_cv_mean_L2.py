from typing import *
import os
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class PlotCVMeanL2(luigi.Task):
	"""
	Luigi Task to plot CV vs mean, indicating selected genes for level 2 clustering
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutL2(tissue=self.tissue, major_class=self.major_class, project=self.project)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".CV_mean.pdf"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			mu = ds.row_attrs["_LogMean"]
			cv = ds.row_attrs["_LogCV"]
			selected = ds.row_attrs["_Selected"].astype('bool')
			excluded = (1 - ds.row_attrs["_Valid"]).astype('bool')

			fig = plt.figure(figsize=(10, 6))
			ax1 = fig.add_subplot(111)
			ax1.scatter(mu, cv, c='grey', marker=".", edgecolors="none")
			ax1.scatter(mu[excluded], cv[excluded], c='red', marker=".", edgecolors="none")
			ax1.scatter(mu[selected], cv[selected], c='blue', marker=".", edgecolors="none")
			fig.savefig(out_file, format="pdf")
			plt.close()
