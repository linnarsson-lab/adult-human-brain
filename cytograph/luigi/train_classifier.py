from typing import *
import os
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi


class classifier(luigi.Config):
	build_dir = luigi.Parameter(default="loom_builds/build_20170114_225636")


class TrainClassifier(luigi.Task):
	"""
	Luigi Task to train a classifier
	"""
	method = luigi.Parameter(default="svc")

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", "classifier.pickle"))

	def run(self) -> None:
		with self.output().temporary_path() as fname:
			logging.info("Retraining classifier")
			clf = cg.Classifier(classifier().build_dir, "mainClass", n_per_cluster=50, use_ica=False, method=self.method)
			tr_fname = os.path.join(classifier().build_dir, "mainClass.loom")
			if not os.path.exists(tr_fname):
				clf.generate()
			ds_training = loompy.connect(tr_fname)
			clf.fit(ds_training)
			with open(fname, "wb") as f:
				pickle.dump(clf, f)
