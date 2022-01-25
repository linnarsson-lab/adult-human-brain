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


def calc_cpu(n_cells):
    n = np.array([1e2, 1e3, 1e4, 1e5, 5e5, 1e6, 2e6])
    cpus = [1, 3, 7, 14, 28, 28, 56]
    idx = (np.abs(n - n_cells)).argmin()
    return cpus[idx]


def split_subset(config, subset: str, method: str = 'coverage', thresh: float = None) -> None:

    loom_file = os.path.join(config.paths.build, "data", subset + ".loom")
    out_dir = os.path.join(config.paths.build, "exported", subset, method)

    with Tempname(out_dir) as exportdir:
        os.mkdir(exportdir)
        with loompy.connect(loom_file) as ds:

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
                total = len(clusters)
                if np.any(np.bincount(clusters) / total < 0.01):
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
                name = chr(i + 65) if i < 26 else chr(i + 39) * 2
                f.write(f'{name}:\n')
                f.write('  include: []\n')
                f.write(f'  onlyif: Split == {i}\n')
                f.write('  execution:\n')
                f.write(f'    n_cpus: {n_cpus}\n')
                f.write(f'    memory: {memory}\n')
    return True
