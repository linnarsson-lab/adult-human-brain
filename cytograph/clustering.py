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
import hdbscan
from sklearn.cluster import DBSCAN


class Clustering:
    def __init__(self, method: str) -> None:
        self.method = method
    
    def fit_predict(self, ds: loompy.LoomConnection) -> np.ndarray:
        n_valid = np.sum(ds.col_attrs["_Valid"] == 1)
        n_total = ds.shape[1]
        logging.info("%d of %d cells were valid", n_valid, n_total)
        logging.info("%d of %d genes were valid", np.sum(ds.row_attrs["_Valid"] == 1), ds.shape[0])
        cells = np.where(ds.col_attrs["_Valid"] == 1)[0]

        if self.method == "hdbscan":
            logging.info("HDBSCAN clustering in t-SNE space")
            min_pts = 10 if n_valid < 3000 else (20 if n_valid < 20000 else 100)
            tsne_pos = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[cells, :]
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_pts)
            labels = clusterer.fit_predict(tsne_pos)
            labels_all = np.zeros(ds.shape[1], dtype='int') + -1
            labels_all[cells] = labels
            ds.set_attr("Clusters", labels_all, axis=1)
        elif self.method == "dbscan":
            logging.info("DBSCAN clustering in t-SNE space")
            min_pts = 10 if n_valid < 3000 else (20 if n_valid < 20000 else 100)
            eps_pct = 65
            tsne_pos = np.vstack((ds.col_attrs["_X"], ds.col_attrs["_Y"])).transpose()[cells, :]
            nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
            nn.fit(tsne_pos)
            knn = nn.kneighbors_graph(mode='distance')
            k_radius = knn.max(axis=1).toarray()
            epsilon = np.percentile(k_radius, eps_pct)
            clusterer = DBSCAN(eps=epsilon, min_samples=min_pts)
            labels = clusterer.fit_predict(tsne_pos)
            labels_all = np.zeros(ds.shape[1], dtype='int') + -1
            labels_all[cells] = labels
            ds.set_attr("Clusters", labels_all, axis=1)
        else:
            logging.info("Louvain clustering on the multiscale KNN graph")
            (a, b, w) = ds.get_edges("KNN", axis=1)
            knn = sparse.coo_matrix((w, (a, b)), shape=(ds.shape[1], ds.shape[1])).tocsr()[cells, :][:, cells]
            lj = cg.LouvainJaccard(resolution=100, jaccard=False)
            labels = lj.fit_predict(knn.tocoo())
            # Make labels for excluded cells == -1
            labels_all = np.zeros(ds.shape[1], dtype='int') + -1
            labels_all[cells] = labels
            ds.set_attr("Clusters", labels_all, axis=1)
        logging.info("Found " + str(max(labels_all) + 1) + " clusters")
        return labels_all
