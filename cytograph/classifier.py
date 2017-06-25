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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
import loompy


class Classifier:
	"""
	Generate test and validation datasets, train a classifier to recognize main classes of cells, then
	split the datasets into new files representing those classes (neurons further split by region).
	"""
	def __init__(self, build_dir: str, n_per_cluster: int, n_genes: int = 2000, n_components: int = 50, use_ica: bool = False) -> None:
		self.build_dir = build_dir
		self.n_per_cluster = n_per_cluster
		self.n_genes = n_genes
		self.n_components = n_components
		self.labels = {}  # type: Dict[str, np.ndarray]
		self.pca = None  # type: cg.PCAProjection
		self.ica = None  # type: FastICA
		self.use_ica = use_ica
		self.mu = None  # type: np.ndarray
		self.classes = None  # type: List[str]
		self.clfs = {}  # type: Dict[str, LogisticRegressionCV]

	def generate(self) -> None:
		"""
		Scan the build folder and generate training dataset
		"""
		foutname = os.path.join(self.build_dir, "classified.loom")
		if os.path.exists(foutname):
			logging.info("Training dataset already exists; reusing it.")
			return
		for fname in os.listdir(self.build_dir):
			if fname.startswith("L0_"):
				ds = loompy.connect(os.path.join(self.build_dir, fname))
				ds_training = None  # type: loompy.LoomConnection
				if os.path.exists(foutname):
					ds_training = loompy.connect(foutname)

				# select cells
				labels = ds.col_attrs["Clusters"]
				temp = []  # type: List[int]
				for i in range(max(labels) + 1):
					temp += list(np.random.choice(np.where(labels == i)[0], size=self.n_per_cluster))
				cells = np.array(temp)
				logging.info("Sampling %d cells from %s", cells.shape[0], fname)
				# put the cells in the training dataset
				for (ix, selection, vals) in ds.batch_scan(cells=cells, axis=1):
					class_labels = ds.col_attrs["SubclassAssigned"][selection]
					if ds_training is None:
						loompy.create(foutname, vals, row_attrs=ds.row_attrs, col_attrs={"SubclassAssigned": class_labels})
						ds_training = loompy.connect(foutname)
					else:
						ds_training.add_columns(vals, {"SubclassAssigned": class_labels})

	def fit(self, ds: loompy.LoomConnection) -> None:
		"""
		Fit, optimize using k-fold cross-validation and then measure the performance. After fitting,
		the predict() method will automatically use the optimal parameters discovered in fitting.
		"""
		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		self.mu = normalizer.mu

		logging.info("Feature selection")
		genes = cg.FeatureSelection(2000).fit(ds)

		logging.info("PCA projection")
		self.pca = cg.PCAProjection(genes, max_n_components=50)
		transformed = self.pca.fit_transform(ds, normalizer)

		self.classes = list(set(ds.col_attrs["SubclassAssigned"]))
		for cls in self.classes:
			self.labels[cls] = (ds.col_attrs["SubclassAssigned"] == cls).astype('int')

			logging.info("Fitting classifier")
			# optimize the classsifier on the training set, then score on the test set
			train_X, test_X, train_Y, test_Y = train_test_split(transformed, self.labels[cls], test_size=0.5, random_state=0)
			self.clfs[cls] = LogisticRegressionCV(Cs=10, solver='sag')
			self.clfs[cls].fit(train_X, train_Y)
			with open(os.path.join(self.build_dir, cls + "_performance.txt"), "w") as f:
				f.write(classification_report(test_Y, self.clfs[cls].predict(test_X), target_names=["Not " + cls, cls]))

	def predict_proba(self, ds: loompy.LoomConnection) -> Tuple[np.ndarray, np.ndarray]:
		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)
		normalizer.mu = self.mu		# Use the same row means as were used during training

		logging.info("PCA projection")
		transformed = self.pca.transform(ds, normalizer)

		logging.info("Class prediction")
		temp = []
		for cls in self.classes:
			temp.append(self.clfs[cls].predict_proba(transformed)[:, 1])
		probs = np.vstack(temp).transpose()
		return (probs, self.classes)

	# def split(self, ds: loompy.LoomConnection, tissue: str, labels: np.ndarray, names: List[str], dsout: Dict[str, loompy.LoomConnection]=None) -> Dict[str, loompy.LoomConnection]:
	# 	if dsout is None:
	# 		dsout = {}
	# 	for (ix, selection, vals) in ds.batch_scan(axis=1):
	# 		for lbl in range(np.max(labels) + 1):
	# 			subset = np.intersect1d(np.where(labels == lbl)[0], selection)
	# 			if subset.shape[0] == 0:
	# 				continue
	# 			m = vals[:, subset - ix]
	# 			ca = {}
	# 			for key in ds.col_attrs:
	# 				ca[key] = ds.col_attrs[key][subset]
	# 			name = names[lbl]
	# 			if name == "Neurons":
	# 				name = name + "_" + tissue
	# 			if name.startswith("Exclude"):
	# 				pass
	# 			if name not in dsout:
	# 				dsout[name] = loompy.create(os.path.join(self.build_dir, "Class_" + name + ".loom"), m, ds.row_attrs, ca)
	# 			else:
	# 				dsout[name].add_columns(m, ca)
	# 	return dsout
