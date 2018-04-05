from typing import *
import numpy as np
import logging
import loompy
import cytograph as cg
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy_groupies as npg
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


class ClusterValidator:
	def __init__(self) -> None:
		self.report: str = None
		self.proba: np.ndarray = None

	def fit(self, ds: loompy.LoomConnection, plot: str = None) -> np.ndarray:
		"""
		Fit a classifier and use it to determine cluster predictive power

		Args:
			ds		Dataset
			plot	Filename for optional plot

		Returns:
			Matrix of classification probabilities, shape (n_cells, n_labels)
		"""
		logging.info("Feature selection")
		nnz = ds.map([np.count_nonzero], axis=0)[0]
		valid_genes = np.logical_and(nnz > 5, nnz < ds.shape[1] * 0.5).astype("int")
		ds.ra._Valid = valid_genes

		logging.info("Normalization")
		normalizer = cg.Normalizer(False)
		normalizer.fit(ds)

		logging.info("Feature selection")
		(_, enrichment, _) = cg.MarkerSelection(findq=False, labels_attr="Clusters").fit(ds)
		genes = np.zeros_like(ds.ra.Gene, dtype=bool)
		for ix in range(enrichment.shape[1]):
			genes[np.argsort(-enrichment[:, ix])[:25]] = True

		logging.info("PCA projection")
		pca = cg.PCAProjection(genes, max_n_components=50)
		transformed = pca.fit_transform(ds, normalizer)

		le = LabelEncoder().fit(ds.ca.ClusterName)
		self.le = le
		labels = le.transform(ds.ca.ClusterName)

		train_X, test_X, train_Y, test_Y = train_test_split(transformed, labels, test_size=0.2)
		classifier = RandomForestClassifier(max_depth=30)
		classifier.fit(train_X, train_Y)
		self.report = classification_report(test_Y, classifier.predict(test_X), target_names=le.classes_)
		self.proba = classifier.predict_proba(transformed)

		if plot:
			agg = npg.aggregate(labels, self.proba, axis=0, func="mean")
			plt.imshow(agg, cmap="viridis")
			plt.xticks(np.arange(le.classes_.shape[0]), le.classes_, rotation="vertical", fontsize=7)
			plt.yticks(np.arange(le.classes_.shape[0]), le.classes_, rotation="horizontal", fontsize=7)
			plt.xlabel("Predicted cell type")
			plt.ylabel("Observed cell type")
			plt.title("Predictive power of cluster identities")
			cbar = plt.colorbar()
			cbar.set_label('Average classification probability', rotation=90)
			plt.savefig(plot, bbox_inches="tight")

		return self.proba
