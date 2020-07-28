
import loompy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def decision_boundary(ds: loompy.LoomConnection, clf, transformed, y_true, score, out_file) -> None:

	plt.figure(None, (10, 10))

	plt.scatter(ds.ca.UMAP3D[:, 0], ds.ca.UMAP3D[:, 1], c='grey', alpha=0.20)
	plt.scatter(transformed[:, 0], transformed[:, 1], c=y_true)

	# # plot the decision function
	# ax = plt.gca()
	# xlim = ax.get_xlim()
	# ylim = ax.get_ylim()
	# # create grid to evaluate model
	# xx = np.linspace(xlim[0], xlim[1], 30)
	# yy = np.linspace(ylim[0], ylim[1], 30)
	# YY, XX = np.meshgrid(yy, xx)
	# xy = np.vstack([XX.ravel(), YY.ravel()]).T
	# Z = clf.decision_function(xy).reshape(XX.shape)
	# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
	# 		linestyles=['--', '-', '--'])
	plt.title(f'Balanced accuracy {score}', fontsize=13)
	plt.axis('off')

	plt.savefig(out_file)
	plt.close()
