import os
import csv
import logging
import numpy as np
import cytograph as cg
from typing import *
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import FastICA, IncrementalPCA
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.naive_bayes import GaussianNB
import loompy


class Classifier:
	"""
	Generate test and validation datasets, train a classifier to recognize main classes of cells, then
	split the datasets into new files representing those classes (neurons further split by region).
	"""
	def __init__(self, build_dir: str, n_per_cluster: int, n_genes: int = 2000, n_components: int = 50, batch_size: int = 1000) -> None:
		self.build_dir = build_dir
		self.n_per_cluster = n_per_cluster
		self.n_genes = n_genes
		self.n_components = n_components
		self.batch_size = batch_size
		self.classes = None  # type: List[str]
		self.labels = None  # type: np.ndarray
		self.pca = None  # type: cg.PCAProjection
		self.mu = None  # type: np.ndarray
		self.sd = None  # type: np.ndarray
		self.classifier = None  # type: SVC
		self.le = None  # type: LabelEncoder

	def generate(self) -> None:
		"""
		Scan the build folder and generate training dataset
		"""
		foutname = os.path.join(self.build_dir, "classified.loom")
		if os.path.exists(foutname):
			logging.info("Training dataset already exists; reusing it.")
			return
		accessions = None  # type: np.ndarray
		for fname in os.listdir(self.build_dir):
			if fname.startswith("L0_"):
				ds = loompy.connect(os.path.join(self.build_dir, fname))
				if accessions is None:
					# Keep track of the gene order in the first file
					accessions = ds.row_attrs["Accession"]
				ds_training = None  # type: loompy.LoomConnection
				if os.path.exists(foutname):
					ds_training = loompy.connect(foutname)

				# select cells
				labels = ds.col_attrs["Clusters"]
				temp = []  # type: List[int]
				for i in range(max(labels) + 1):
					candidate_cells = np.where(labels == i)[0]
					if ds.col_attrs["SubclassAssigned"][candidate_cells[0]] == "Unknown":
						continue
					if candidate_cells.shape[0] > self.n_per_cluster:
						temp += list(np.random.choice(candidate_cells, size=self.n_per_cluster, replace=False))
					else:
						temp += list(candidate_cells)
				cells = np.array(sorted(temp))
				logging.info("Sampling %d cells from %s", cells.shape[0], fname)

				# put the cells in the training dataset
				# This is magic sauce for making the order of one list be like another
				ordering = np.where(ds.row_attrs["Accession"][None, :] == accessions[:, None])[1]
				for (ix, selection, vals) in ds.batch_scan(cells=cells, axis=1, batch_size=self.batch_size):
					ca = {key: val[selection] for key, val in ds.col_attrs.items()}
					if ds_training is None:
						loompy.create(foutname, vals, row_attrs=ds.row_attrs, col_attrs=ca)
						ds_training = loompy.connect(foutname)
					else:
						ds_training.add_columns(vals[ordering, :], ca)

	def fit(self, ds: loompy.LoomConnection) -> None:
		"""
		Fit, optimize using k-fold cross-validation and then measure the performance. After fitting,
		the predict() method will automatically use the optimal parameters discovered in fitting.
		"""
		logging.info("Normalization")
		normalizer = cg.Normalizer(True)
		normalizer.fit(ds)
		self.mu = normalizer.mu
		self.sd = normalizer.sd

		logging.info("Feature selection")
		genes = cg.FeatureSelection(2000).fit(ds)

		logging.info("PCA projection")
		self.pca = cg.PCAProjection(genes, max_n_components=50, batch_size=self.batch_size)
		transformed = self.pca.fit_transform(ds, normalizer)

		self.classes = ds.col_attrs["SubclassAssigned"]
		self.le = LabelEncoder().fit(self.classes)
		self.labels = self.le.transform(self.classes)

		train_X, test_X, train_Y, test_Y = train_test_split(transformed, self.labels, test_size=0.2, random_state=0)
		important_classes = [
			"Astrocyte",
			"Astrocyte,Cycling",
			"Bergmann-glia",
			"Blood",
			"Blood,Cycling",
			"Ependymal",
			"Immune",
			"Neurons",
			"Neurons,Cycling",
			"OEC",
			"Oligos",
			"Oligos,Cycling",
			"Satellite-glia",
			"Satellite-glia,Cycling",
			"Schwann",
			"Ttr",
			"Vascular",
			"Vascular,Cycling"
		]
		self.classifier = SVC(class_weight={c: 10 for c in self.le.transform(important_classes)})
		# self.classifier = SGDClassifier(loss="log")
		self.classifier.fit(train_X, train_Y)
		with open(os.path.join(self.build_dir, "performance.txt"), "w") as f:
			f.write(classification_report(test_Y, self.classifier.predict(test_X), target_names=self.le.classes_))

	def predict(self, ds: loompy.LoomConnection) -> List[str]:
		logging.info("Normalization")
		normalizer = cg.Normalizer(True)
		normalizer.fit(ds)
		normalizer.mu = self.mu		# Use the same row means as were used during training
		normalizer.sd = self.sd

		logging.info("PCA projection")
		transformed = self.pca.transform(ds, normalizer)

		logging.info("Class prediction")
		labels = self.classifier.predict(transformed)
		return self.le.inverse_transform(labels)
