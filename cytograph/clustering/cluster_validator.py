import matplotlib.pyplot as plt
import numpy as np
import numpy_groupies as npg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import loompy
from cytograph.decomposition import HPF


class ClusterValidator:
	def __init__(self) -> None:
		self.report: str = ""
		self.proba: np.ndarray = None

	def fit(self, ds: loompy.LoomConnection, plot_file: str = None, report_file: str = None) -> np.ndarray:
		"""
		Fit a classifier and use it to determine cluster predictive power

		Args:
			ds		Dataset
			plot_file	Filename for optional plot
			report_file	Filename for optional report

		Returns:
			Matrix of classification probabilities, shape (n_cells, n_labels)
		"""

		if "ClusterName" in ds.ca:
			cluster_names = [str(ds.ca.ClusterName[ds.ca.Clusters == lbl][0]) for lbl in np.unique(ds.ca.Clusters)]
		else:
			cluster_names = [str(lbl) for lbl in np.unique(ds.ca.Clusters)]
		
		genes = np.where(ds.ra.Selected==1)[0]
		data = ds.sparse(rows=genes).T
		hpf = cg.HPF(k=ds.ca.HPF.shape[1], validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
		hpf.fit(data)
		theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T

		train_X, test_X, train_Y, test_Y = train_test_split(theta, ds.ca.Clusters, test_size=0.2)
		classifier = RandomForestClassifier(max_depth=30)
		classifier.fit(train_X, train_Y)
		self.report = classification_report(test_Y, classifier.predict(test_X), labels=np.unique(ds.ca.Clusters), target_names=cluster_names)
		self.proba = classifier.predict_proba(theta)

		if plot_file is not None:
			plt.figure()
			agg = npg.aggregate(ds.ca.Clusters, self.proba, axis=0, func="mean")
			plt.imshow(agg, cmap="viridis")
			plt.xticks(np.arange(len(cluster_names)), cluster_names, rotation="vertical", fontsize=7)
			plt.yticks(np.arange(len(cluster_names)), cluster_names, rotation="horizontal", fontsize=7)
			plt.xlabel("Predicted cluster")
			plt.ylabel("Ground truth cluster")
			plt.title("Cluster quality (predictive power)")
			cbar = plt.colorbar()
			cbar.set_label('Probability of predicted cluster', rotation=90)
			plt.savefig(plot_file, bbox_inches="tight")
			plt.close()
		if report_file is not None:
			with open(report_file, "w") as f:
				f.write(self.report)

		return self.proba
