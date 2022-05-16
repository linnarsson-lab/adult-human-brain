import os
import logging
import loompy
import numpy as np
from scipy.cluster.hierarchy import cut_tree
import matplotlib.pyplot as plt
from ..plotting.colors import colorize
from ..pipeline import Tempname
import networkx as nx
import community
from sknetwork.hierarchy import Paris


def calc_cpu(n_cells):
    n = np.array([1e2, 1e3, 1e4, 1e5, 5e5, 1e6, 2e6])
    cpus = [1, 3, 7, 14, 28, 28, 56]
    idx = (np.abs(n - n_cells)).argmin()
    return cpus[idx]

def coverage(clusters, G):
    partition = [set(np.where(clusters == c)[0]) for c in np.unique(clusters)]
    return nx.community.partition_quality(G, partition)[0]

def split_subset(config, subset: str, method: str = 'coverage', thresh: float = None) -> None:
    """
    split
    """
    loom_file = os.path.join(config.paths.build, "data", subset + ".loom")
    out_dir = os.path.join(config.paths.build, "exported", subset, method)

    with Tempname(out_dir) as exportdir:
        os.mkdir(exportdir)
        with loompy.connect(loom_file) as ds:

            if ds.ca.Clusters.max() == 0:
                return False

            if method == 'dendrogram':

                logging.info("Splitting by dendrogram")

                # split dendrogram into two and get new clusters
                agg_file = os.path.join(config.paths.build, "data", subset + ".agg.loom")
                with loompy.connect(agg_file, 'r') as dsagg:
                    Z = dsagg.attrs.linkage
                    branch = cut_tree(Z, 2).T[0]
                clusters = branch[ds.ca.Clusters]

                # save split attribute and plot
                ds.ca.Split = clusters
                plt.figure(None, (16, 16))
                plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Split), s=5)
                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

            if method == 'coverage':

                # set thresh to 0.98 if not specified
                if thresh is None:
                    thresh = 0.98

                logging.info(f"Splitting by dendrogram if coverage is above {thresh}")

                # Get dendrogram from agg file and split into two
                logging.info("Splitting dendrogram in .agg file")
                agg_file = os.path.join(config.paths.build, "data", subset + ".agg.loom")
                with loompy.connect(agg_file, 'r') as dsagg:
                    Z = dsagg.attrs.linkage
                    branch = cut_tree(Z, 2).T[0]

                # Assign clusters based on the dendrogram cut
                clusters = branch[ds.ca.Clusters]

                # Check cluster sizes
                if np.any(np.bincount(clusters) < 25):
                    logging.info(f"A cluster is too small.")
                    return False

                # Load KNN graph
                logging.info("Loading KNN graph")
                G = nx.from_scipy_sparse_matrix(ds.col_graphs.KNN)

                # Calculate coverage of this partition on the graph
                logging.info("Calculating coverage of this partition")
                partition = []
                for c in np.unique(clusters):
                    partition.append(set(np.where(clusters == c)[0]))
                cov = nx.algorithms.community.quality.coverage(G, partition)

                # Stop if coverage is below thresh
                ds.attrs.Coverage = cov
                logging.info(f"Coverage threshold set at {thresh}")
                if cov < thresh:
                    logging.info(f"Partition is not separable: {cov:.5f}.")
                    return False

                # Otherwise save split attribute and plot
                ds.ca.Split = clusters
                logging.info(f"Partition is separable: {cov:.5f}.")
                logging.info(f"Plotting partition")
                plt.figure(None, (16, 16))
                plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Split), s=5)
                plt.axis('off')
                plt.title(f"Coverage: {cov:.5f}", fontsize=20)
                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

            if method == 'paris':

                    # change thresh to 0.999 if not specified
                    if thresh is None:
                        thresh = 0.999

                    # load, partition, and cluster graph
                    logging.info("Loading KNN graph")
                    G = nx.from_scipy_sparse_matrix(ds.col_graphs.KNN)
                    logging.info("Partitioning graph by Cytograph clusters")
                    partition = dict(zip(np.arange(ds.shape[1]), ds.ca.Clusters))
                    logging.info("Generating induced adjacency  matrix")
                    induced = community.induced_graph(partition, G)
                    adj = nx.linalg.graphmatrix.adjacency_matrix(induced)
                    logging.info("Paris clustering")
                    Z = Paris().fit_transform(adj)
                    ds.attrs.paris_linkage = Z

                    # calculate coverage for a range of cuts
                    logging.info("Calculating coverage")
                    cov = [coverage(cut_tree(Z, n).T[0][ds.ca.Clusters], G) for n in range(2, 11)]
                    cov = np.array(cov)

                    # Plot coverage
                    plt.figure(None, (5, 4))
                    plt.scatter(range(2, 11), cov)
                    plt.hlines(thresh, 2, 10)
                    plt.xticks(range(2, 11))
                    plt.title('Coverage', fontsize=15)
                    plt.ylabel('Coverage')
                    plt.xlabel('Number of Cuts')
                    plt.tight_layout()
                    plt.savefig(os.path.join(exportdir, "Coverage.png"), dpi=150)
                    plt.close()

                    # Check coverage
                    logging.info(f"Coverage threshold set at {thresh}")
                    if not any(cov > thresh):
                        logging.info("No split found")
                        clusters = np.zeros(ds.shape[1])
                        return False
                    else:
                        n = np.where(cov > thresh)[0][-1] + 2
                        logging.info(f"Cutting Paris dendrogram into {n} branches")
                        branch = cut_tree(Z, n).T[0]
                        clusters = branch[ds.ca.Clusters]

                    # Otherwise save split attribute and plot
                    ds.ca.Split = clusters
                    plt.figure(None, (4, 4))
                    plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Split), s=1)
                    plt.title('Split')
                    plt.axis('off')
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
                # Calc cpu and memory
                n_cpus = calc_cpu(sizes[i])
                memory = 750 if n_cpus == 56 else config.execution.memory
                # Write to punchcard
                name = chr(i % 26 + 65) * (1 + int(i / 26))
                f.write(f'{name}:\n')
                f.write('  include: []\n')
                f.write(f'  onlyif: Split == {i}\n')
                f.write('  execution:\n')
                f.write(f'    n_cpus: {n_cpus}\n')
                f.write(f'    memory: {memory}\n')
    return True
