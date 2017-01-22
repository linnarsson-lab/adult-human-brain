import os
import csv
import logging
import numpy as np
import cytograph as cg
from typing import *
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import FastICA, IncrementalPCA
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import loompy


class Classifier:
	"""
	Generate test and validation datasets, train a classifier to recognize main classes of cells, then
	split the datasets into new files representing those classes (neurons further split by region).
	"""
	def __init__(self, build_dir: str, classes: str, n_per_cluster: int, n_genes: int = 2000, n_components: int = 50) -> None:
		self.build_dir = build_dir
		self.n_per_cluster = n_per_cluster
		self.ds_training = None  # type: loompy.LoomConnection
		self.ds_validation = None  # type: loompy.LoomConnection
		self.classes = classes
		self.n_genes = n_genes
		self.n_components = n_components
		self.naive_bayes = None  # type: GaussianNB
		self.label_encoder = None  # type: LabelEncoder
		self.pca = None  # type: cg.PCAProjection
		self.ica = None  # type: FastICA

	def generate(self) -> None:
		"""
		Scan the build folder and generate training and validation datasets
		"""
		for f in os.listdir(self.build_dir):
			if f.endswith(self.classes + ".txt"):
				tissue = f.split("_")[0]
				# Load the class definitions
				logging.info("Loading class definitions for " + f)
				class_defs = {}
				with open(os.path.join(self.build_dir, f), mode='r') as infile:
					reader = csv.reader(infile)
					for row in reader:
						if len(row) == 2:
							class_defs[row[0]] = row[1]
							logging.info(row[0] + ": " + row[1])
				self.ds_training = self._generate_samples_for_file(tissue, class_defs, "training")
				self.ds_validation = self._generate_samples_for_file(tissue, class_defs, "validation")

	def _generate_samples_for_file(self, tissue: str, class_defs: Dict[int, str], dataset_name: str) -> loompy.LoomConnection:
		"""
		Add to a classification dataset (training, test, validation) from one particular tissue
		"""
		logging.info("Generating samples for " + tissue + " (" + dataset_name + ")")
		ds = loompy.connect(os.path.join(self.build_dir, tissue + ".loom"))
		dsout = None  # type: loompy.LoomConnection

		# select cells
		labels = ds.col_attrs["Clusters"]
		temp = []  # type: List[int]
		for i in range(max(labels) + 1):
			temp += list(np.random.choice(np.where(labels == i)[0], size=self.n_per_cluster))
		cells = np.array(temp)
		logging.info("Sampled %d cells for %s (%s)", cells.shape[0], tissue, dataset_name)
		# put the cells in the training and validation datasets
		for (ix, selection, vals) in ds.batch_scan(cells=cells, axis=1):
			class_labels = np.empty(selection.shape[0], dtype='object')
			for key in class_defs:
				class_labels[labels[selection] == key] = class_defs[key]
			class_labels = class_labels.astype(str)
			if dsout is None:
				fname = os.path.join(self.build_dir, self.classes + "_" + dataset_name + ".loom")
				loompy.create(fname, vals, row_attrs=ds.row_attrs, col_attrs={"Class": class_labels})
				dsout = loompy.connect(fname)
			else:
				dsout.add_columns(vals, {"Class": class_labels})

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)

		logging.info("Feature selection")
		genes = cg.FeatureSelection(2000).fit(ds)

		logging.info("PCA projection")
		self.pca = cg.PCAProjection(genes, max_n_components=50)
		pca_transformed = self.pca.fit_transform(ds, normalizer)

		logging.info("FastICA projection")
		self.ica = FastICA()
		ica_transformed = self.ica.fit_transform(pca_transformed)

		logging.info("Naïve Bayes")
		true_labels = ds.col_attrs["Class"]
		self.label_encoder = LabelEncoder()
		self.label_encoder.fit(list(set(true_labels)))
		self.naive_bayes = GaussianNB()
		self.naive_bayes.fit(ica_transformed, self.label_encoder.transform(true_labels))

	def predict(self, ds: loompy.LoomConnection) -> np.ndarray:
		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)

		logging.info("PCA projection")
		pca_transformed = self.pca.fit_transform(ds, normalizer)

		logging.info("FastICA projection")
		self.ica = FastICA()
		ica_transformed = self.ica.fit_transform(pca_transformed)

		logging.info("Class prediction by Naïve Bayes")
		labels = self.naive_bayes.predict(ica_transformed)

		if "Class" in ds.col_attrs:
			true_labels = self.label_encoder.transform(ds.col_attrs["Class"])
			logging.info(classification_report(true_labels, labels, self.label_encoder.classes_))

		return self.label_encoder.inverse_transform(labels)

	def split(self, ds: loompy.LoomConnection, tissue: str, labels: List[str], split_def: Dict[str, str]) -> None:
		pass
