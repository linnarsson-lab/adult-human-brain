import os
import logging
import loompy
import numpy as np
import numpy_groupies as npg
from sklearn import svm
from sklearn.metrics import recall_score
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import cut_tree
import matplotlib.pyplot as plt
from ..clustering import Louvain
from ..plotting.colors import colorize
from ..plotting import decision_boundary
from ..pipeline import Tempname
import networkx as nx
from sknetwork.hierarchy import cut_straight, Paris
import community
import time

def calc_cpu(n_cells):
    x = np.log10([1, 10, 100, 1e3, 1e4, 1e5, 1e6])
    y = [1, 1, 1, 7, 14, 28, 56]
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    cpus = max(1, int(p(np.log10(n_cells))))
    return min(cpus, 56)

def split_subset(config, subset: str, method: str = 'coverage', thresh: float = 0.99) -> None:
    """
    Uses support vector classification to find separable clusters on the UMAP
    """

    loom_file = os.path.join(config.paths.build, "data", subset + ".loom")
    out_dir = os.path.join(config.paths.build, "exported", subset, method)

    with Tempname(out_dir) as exportdir:
        os.mkdir(exportdir)
        with loompy.connect(loom_file) as ds:

            # Stop if only one cluster is left
            if ds.ca.Clusters.max() == 0:
                logging.info("Only one cluster found.")
                return False

            # change method to dendrogram if more than # of clusters
            if method == 'dendrogram':

                logging.info("Splitting by dendrogram")
                # split dendrogram into two and get new clusters
                agg_file = os.path.join(config.paths.build, "data", subset + ".agg.loom")
                with loompy.connect(agg_file, 'r') as dsagg:
                    Z = dsagg.attrs.linkage
                    branch = cut_tree(Z, 2).T[0]
                clusters = np.array([branch[x] for x in ds.ca.Clusters])
                # plot split
                ds.ca.Split = clusters
                plt.figure(None, (16, 16))
                plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Split), s=5)
                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

            if method == 'coverage':

                logging.info("Splitting by dendrogram if coverage is above threshold")

                # load KNN graph
                logging.info("Loading KNN graph")
                G = nx.from_scipy_sparse_matrix(ds.col_graphs.KNN)

                # Split dendrogram into two and get new clusters
                # possibly to be replaced with Paris clustering on the KNN
                logging.info("Splitting dendrogram in .agg file")
                agg_file = os.path.join(config.paths.build, "data", subset + ".agg.loom")
                with loompy.connect(agg_file, 'r') as dsagg:
                    Z = dsagg.attrs.linkage
                    branch = cut_tree(Z, 2).T[0]
                clusters = np.array([branch[x] for x in ds.ca.Clusters])

                # Calculate coverage of this partition on the graph
                logging.info("Calculating coverage of this partition")
                partition = []
                for c in np.unique(clusters):
                    partition.append(set(np.where(clusters == c)[0]))
                cov = nx.algorithms.community.quality.coverage(G, partition)

                # Stop if coverage is below thresh
                if cov < thresh:
                    logging.info(f"Partition is not separable: {cov:.5f}.")
                    return False

                # Otherwise save and plot separable clusters
                logging.info(f"Partition is separable: {cov:.5f}.")
                logging.info(f"Plotting partition")
                _, clusters = np.unique(clusters, return_inverse=True)
                ds.ca.Split = clusters
                plt.figure(None, (16, 16))
                plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Split), s=5)
                plt.title(f"Coverage: {cov:.5f}")
                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

            if method == 'cluster':

                logging.info("Splitting by clusters.")
                clusters = ds.ca.Clusters
                ds.ca.Split = clusters

        # Calculate split sizes
        sizes = np.bincount(clusters)
        logging.info("Creating punchcard")
        with open(f'punchcards/{subset}.yaml', 'w') as f:
            for i in np.unique(clusters):
                # Calc cpu usage
                n_cpus = calc_cpu(sizes[i])
                if n_cpus > 50:
                    memory = 750
                else:
                    memory = config.execution.memory
                # Write to punchcard
                name = chr(i + 65) if i < 26 else chr(i + 39) * 2
                f.write(f'{name}:\n')
                f.write('  include: []\n')
                f.write(f'  onlyif: Split == {i}\n')
                if sizes[i] <= 50:
                    f.write('  params:\n')
                    f.write(f'    k: {int(sizes[i] / 3)}\n')
                    f.write(f'    features: variance\n')
                elif sizes[i] <= 1000:
                    f.write(f'  steps: nn, embeddings, clustering, aggregate, export\n')
                f.write('  execution:\n')
                f.write(f'    n_cpus: {n_cpus}\n')
                f.write(f'    memory: {memory}\n')

    return True
