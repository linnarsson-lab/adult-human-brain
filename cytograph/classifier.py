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
	def __init__(self, classified_dir: str, n_per_cluster: int, n_genes: int = 2000, n_components: int = 50) -> None:
		self.classified_dir = classified_dir
		self.n_per_cluster = n_per_cluster
		self.n_genes = n_genes
		self.n_components = n_components
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
		foutname = os.path.join(self.classified_dir, "classified.loom")
		if os.path.exists(foutname):
			logging.info("Training dataset already exists; reusing it.")
			return
		accessions = None  # type: np.ndarray
		for fname in os.listdir(self.classified_dir):
			if fname.startswith("L1_") and fname.endswith(".loom"):
				ds = loompy.connect(os.path.join(self.classified_dir, fname))
				afname = fname[:-5] + "_10-Jul-2017_clusters_mainClass.txt"
				if os.path.exists(os.path.join(self.classified_dir, afname)):
					logging.info("Loading subclass assignments from " + afname)
					d = {}
					with open(os.path.join(self.classified_dir, afname), "r") as f:
						for line in f.readlines():
							items = line[:-1].split("\t")
							d[int(items[0])] = items[1]
					sa = np.array(list(map(lambda x: d[x] if d[x] != "Outliers" else "Unknown", ds.Clusters)))
					ds.set_attr("SubclassAssigned", sa, axis=1)
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
				for (ix, selection, vals) in ds.batch_scan(cells=cells, axis=1, batch_size=cg.memory().axis1):
					ca = {key: val[selection] for key, val in ds.col_attrs.items()}
					if ds_training is None:
						loompy.create(foutname, vals[ordering, :], row_attrs=ds.row_attrs, col_attrs=ca)
						ds_training = loompy.connect(foutname)
					else:
						ds_training.add_columns(vals[ordering, :], ca)

		# We had a bug after newly creating a loom file, so close and reopen to be sure it's flushed
		ds_training.close()
		ds_training = loompy.connect(foutname)

		# Make sure we don't have both "Neurons,Oligos" and "Oligos,Neurons"
		classes = ds_training.col_attrs["SubclassAssigned"]
		classes_fixed = []
		for cls in classes:
			items = cls.split(",")
			if len(items) == 2 and items[1] != "Cycling":
				items = sorted(items)
			classes_fixed.append(",".join(items))
		ds_training.set_attr("SubclassAssigned", np.array(classes_fixed), axis=1)

	def aggregate_export(self) -> None:
		# Aggregate and compute enrichment, trinaries etc.
		logging.info("Aggregating loom file")
		ds_training = loompy.connect(os.path.join(self.classified_dir, "classified.loom"))
		classes = ds_training.col_attrs["SubclassAssigned"]
		ds_training.set_attr("Clusters", LabelEncoder().fit_transform(classes), axis=1)
		out_file = os.path.join(self.classified_dir, "classified.agg.loom")
		cg.Aggregator(10).aggregate(ds_training, out_file)
		dsagg = loompy.connect(out_file)

		logging.info("Computing auto-annotation")
		aa = cg.AutoAnnotator(root=cg.paths().autoannotation)
		aa.annotate_loom(dsagg)
		aa.save_in_loom(dsagg)

		logging.info("Computing auto-auto-annotation")
		n_clusters = dsagg.shape[1]
		(selected, selectivity, specificity, robustness) = cg.AutoAutoAnnotator(n_genes=6).fit(dsagg)
		dsagg.set_attr("MarkerGenes", np.array([" ".join(ds_training.Gene[selected[:, ix]]) for ix in np.arange(n_clusters)]), axis=1)
		np.set_printoptions(precision=1, suppress=True)
		dsagg.set_attr("MarkerSelectivity", np.array([str(selectivity[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
		dsagg.set_attr("MarkerSpecificity", np.array([str(specificity[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
		dsagg.set_attr("MarkerRobustness", np.array([str(robustness[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
		dsagg.close()

		out_dir = os.path.join(self.classified_dir, "classified_exported")
		logging.info("Exporting cluster data")
		if not os.path.exists(out_dir):
			os.mkdir(out_dir)
		dsagg = loompy.connect(out_file)
		dsagg.export(os.path.join(out_dir, "classified_expression.tab"))
		dsagg.export(os.path.join(out_dir, "classified_enrichment.tab"), layer="enrichment")
		dsagg.export(os.path.join(out_dir, "classified_enrichment_q.tab"), layer="enrichment_q")
		dsagg.export(os.path.join(out_dir, "classified_trinaries.tab"), layer="trinaries")

	def fit(self, ds: loompy.LoomConnection) -> None:
		# Validating genes
		logging.info("Marking invalid genes")
		nnz = ds.map([np.count_nonzero], axis=0)[0]
		valid_genes = np.logical_and(nnz > 5, nnz < ds.shape[1] * 0.5).astype("int")
		ds.set_attr("_Valid", valid_genes, axis=0)
		with open(os.path.join(self.classified_dir, "genes.txt"), "w") as f:
			for ix in range(valid_genes.shape[0]):
				f.write(ds.Accession[ix])
				f.write("\t")
				f.write(str(valid_genes[ix]))
				f.write("\n")

		logging.info("Normalization")
		normalizer = cg.Normalizer(True)
		normalizer.fit(ds)
		self.mu = normalizer.mu
		self.sd = normalizer.sd

		logging.info("Feature selection")
		genes = cg.FeatureSelection(2000).fit(ds)

		logging.info("PCA projection")
		self.pca = cg.PCAProjection(genes, max_n_components=50)
		transformed = self.pca.fit_transform(ds, normalizer)

		self.classes = ds.col_attrs["SubclassAssigned"]
		self.le = LabelEncoder().fit(self.classes)
		self.labels = self.le.transform(self.classes)

		train_X, test_X, train_Y, test_Y = train_test_split(transformed, self.labels, test_size=0.2, random_state=0)
		self.classifier = SVC(probability=True)
		self.classifier.fit(train_X, train_Y)
		with open(os.path.join(self.classified_dir, "performance.txt"), "w") as f:
			f.write(classification_report(test_Y, self.classifier.predict(test_X), target_names=self.le.classes_))
		
	def predict(self, ds: loompy.LoomConnection, probability: bool = False) -> Union[List[str], Tuple[List[str], np.ndarray, List[str]]]:
		logging.info("Normalization")
		normalizer = cg.Normalizer(True)
		normalizer.fit(ds)
		normalizer.mu = self.mu		# Use the same row means as were used during training
		normalizer.sd = self.sd

		logging.info("PCA projection")
		transformed = self.pca.transform(ds, normalizer)

		logging.info("Class prediction")
		labels = self.classifier.predict(transformed)
		if probability == False:
			return self.le.inverse_transform(labels)
		else:
			probs = self.classifier.predict_proba(transformed)
			return (self.le.inverse_transform(labels), probs, self.le.inverse_transform(self.classifier.classes_))
