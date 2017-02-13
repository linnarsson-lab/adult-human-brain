from typing import *
import os
from shutil import copyfile
import numpy as np
import logging
import luigi
import cytograph as cg
import loompy
import logging
from scipy import sparse
from scipy.special import polygamma
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.decomposition import PCA, IncrementalPCA, FastICA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import BallTree, NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import scale
from sklearn.svm import SVR
from scipy.stats import ks_2samp
import networkx as nx


# Config classes should be camel cased
class clustering(luigi.Config):
	n_genes = luigi.IntParameter(default=2000)
	standardize = luigi.BoolParameter(default=False)
	n_components = luigi.IntParameter(default=50)
	use_ica = luigi.BoolParameter(default=True)
	k = luigi.IntParameter(default=30)
	lj_resolution = luigi.FloatParameter(default=1.0)
	layout_method = luigi.Parameter(default='sfdp')


def cluster_layout(ds: loompy.LoomConnection) -> None:
	n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
	n_total = ds.shape[1]
	logging.info("%d of %d cells were valid", n_valid, n_total)
	logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])
	cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

	logging.info("Normalization")
	normalizer = cg.Normalizer(clustering().standardize)
	normalizer.fit(ds)
	ds.set_attr("_LogCV", np.log(normalizer.sd) - np.log(normalizer.mu))
	ds.set_attr("_LogMean", np.log(normalizer.mu))

	logging.info("Selecting %d genes", clustering().n_genes)
	genes = cg.FeatureSelection(clustering().n_genes).fit(ds)
	temp = np.zeros(ds.shape[0])
	temp[genes] = 1
	ds.set_attr("_Selected", temp, axis=0)

	logging.info("PCA projection")
	pca = cg.PCAProjection(genes, max_n_components=clustering().n_components)
	pca_transformed = pca.fit_transform(ds, normalizer, cells=cells)
	transformed = pca_transformed

	if clustering().use_ica:
		logging.info("FastICA projection")
		transformed = FastICA().fit_transform(pca_transformed)

	logging.info("Generating KNN graph")
	knn = kneighbors_graph(transformed, mode='distance', n_neighbors=clustering().k)
	knn = knn.tocoo()
	ds.set_edges("KNN", cells[knn.row], cells[knn.col], knn.data, axis=1)
	mknn = knn.minimum(knn.transpose()).tocoo()
	ds.set_edges("MKNN", cells[mknn.row], cells[mknn.col], mknn.data, axis=1)

	logging.info("Louvain-Jaccard clustering")
	lj = cg.LouvainJaccard(resolution=clustering().lj_resolution)
	labels = lj.fit_predict(knn)
	# Make labels for excluded cells == -1
	labels_all = np.zeros(ds.shape[1], dtype='int') + -1
	labels_all[cells] = labels
	ds.set_attr("Clusters", labels_all, axis=1)

	if clustering().layout_method == "sfdp":
		logging.info("SFDP layout")
		sfdp_pos = cg.SFDP().layout(lj.graph)
		sfdp_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(sfdp_pos, axis=0)
		sfdp_all[cells] = sfdp_pos
		ds.set_attr("_X", sfdp_all[:, 0], axis=1)
		ds.set_attr("_Y", sfdp_all[:, 1], axis=1)
	else:
		logging.info("TSNE layout")
		tsne_pos = cg.TSNE().layout(transformed)
		tsne_all = np.zeros((ds.shape[1], 2), dtype='int') + np.min(tsne_pos, axis=0)
		tsne_all[cells] = tsne_pos
		ds.set_attr("_X", tsne_all[:, 0], axis=1)
		ds.set_attr("_Y", tsne_all[:, 1], axis=1)

