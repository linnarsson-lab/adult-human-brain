from typing import *
import os
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class PlotCVMeanL1(luigi.Task):
	"""
	Luigi Task to plot CV vs mean, indicating selected genes
	"""
	tissue = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.ClusterLayoutL1(tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.tissue + ".LJ.CV_mean.pdf"))

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			mu = ds.row_attrs["_LogCV"]
			cv = ds.row_attrs["_LogMean"]
			selected = ds.row_attrs["_Selected"].astype('bool')
			excluded = (1 - ds.row_attrs["_Valid"]).astype('bool')

			fig = plt.figure(figsize=(10, 6))
			ax1 = fig.add_subplot(111)
			ax1.scatter(mu, cv, c='grey', marker=".", edgecolors="none")
			ax1.scatter(mu[excluded], cv[excluded], c='red', marker=".", edgecolors="none")
			ax1.scatter(mu[selected], cv[selected], c='blue', marker=".", edgecolors="none")
			fig.savefig(out_file, format="pdf")
			plt.close()
