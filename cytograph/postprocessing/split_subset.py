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


def calc_cpu(n_cells):
    x = np.log10([1, 10, 100, 1e3, 1e4, 1e5, 1e6])
    y = [1, 1, 1, 7, 14, 28, 56]
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    cpus = max(1, int(p(np.log10(n_cells))))
    return min(cpus, 56)


def separable(ds, clusters, a, b, exportdir=None):
    """
    Tests if two clusters are separable on the UMAP
    """
    a_ix = np.where(clusters == a)[0]
    b_ix = np.where(clusters == b)[0]
    # subsample if necessary
    sizes = len(a_ix), len(b_ix)
    a_ix = np.random.choice(a_ix, size=5000, replace=False) if sizes[0] > 5000 else a_ix
    b_ix = np.random.choice(b_ix, size=5000, replace=False) if sizes[1] > 5000 else b_ix
    cells = np.hstack((a_ix, b_ix))
    # fit LinearSVC
    transformed = ds.ca.UMAP3D[cells, :]
    y_true = np.ones(cells.shape[0])
    y_true[clusters[cells] == a] = 0
    # Do not regularize
    clf = svm.LinearSVC(C=0.1, dual=False, max_iter=5000)
    clf.fit(transformed, y_true)
    y_pred = clf.predict(transformed)
    # Calculate balanced accuracy score and plot
    score = np.min(recall_score(y_true, y_pred, average=None))
    logging.info(f'{a}, {b}: Score is {score}')
    if exportdir:
        decision_boundary(ds, clf, transformed, y_true, score, os.path.join(exportdir, f'{ds.ca.ClustersUnpolished[clusters == a][0]}.png'))

    return score > 0.99


def split_subset(config, subset: str, method: str) -> None:
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

            if method == 'svc':

                # Recluster data without polishing on the manifold
                logging.info("Reclustering without polish")
                clusters = Louvain(graph="RNN", embedding="UMAP").fit_predict(ds)
                ds.ca.ClustersUnpolished = clusters

                # Plot unpolished clusters
                logging.info("Plotting ClustersUnpolished")
                plt.figure(None, (16, 16))
                plt.scatter(ds.ca.UMAP3D[:, 0], ds.ca.UMAP3D[:, 1], c=colorize(ds.ca.ClustersUnpolished), s=5)
                plt.savefig(os.path.join(exportdir, "ClustersUnpolished.png"), dpi=150)
                plt.close()

                logging.info("Testing for separable clusters with SVC")

                clusters = np.copy(ds.ca.ClustersUnpolished)
                # Track separable clusters
                separable_clusters = []
                for c1 in np.unique(ds.ca.ClustersUnpolished):
                    # Skip c1 if it was already merged
                    if not np.isin(c1, clusters):
                        continue
                    merge_flag = True
                    while merge_flag:
                        # Stop if only one cluster or at the last cluster
                        if len(np.unique(clusters)) == 1 or c1 == clusters.max():
                            merge_flag = False
                            break
                        # Test if cluster is separable by SVC
                        one_vs_all = np.ones(ds.shape[1])
                        one_vs_all[clusters == c1] = 0
                        if separable(ds, one_vs_all, 0, 1, exportdir):
                            logging.info(f'{c1} is separable')
                            merge_flag = False
                            separable_clusters.append(c1)
                            break
                        else:
                            logging.info(f'{c1} is not separable')
                            # Calculate distances between remaining clusters
                            mu = npg.aggregate(clusters, ds.ca.UMAP.T, func='mean', axis=1, fill_value=0)
                            ix = np.unique(clusters)
                            D = squareform(pdist(mu[:, ix].T, 'euclidean'))
                            # Map index of D to cluster
                            mapping = dict(zip(range(len(ix)), ix))
                            # Select row of D that corresponds to c1
                            for temp in np.argsort(D[np.where(ix == c1)[0][0], :]):
                                # Find c2 corresponding to column of D
                                c2 = mapping[temp]
                                if (c1 != c2) and not (c2 in separable_clusters):
                                    logging.info(f'Testing {c1} and {c2}')
                                    if not separable(ds, clusters, c1, c2, exportdir):
                                        logging.info(f'Merging cluster {c2} into cluster {c1}')
                                        clusters[(clusters == c1) | (clusters == c2)] = c1
                                        merge_flag = True
                                        break
                                    else:
                                        merge_flag = False
                            # Cluster is separable if unable to merge with any clusters
                            logging.info(f'{c1} is separable')
                            separable_clusters.append(c1)

                # Finalize, save, plot separable clusters
                _, clusters = np.unique(clusters, return_inverse=True)
                ds.ca.Split = clusters
                plt.figure(None, (16, 16))
                plt.scatter(ds.ca.UMAP3D[:, 0], ds.ca.UMAP3D[:, 1], c=colorize(ds.ca.Split), s=5)
                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

                # Stop if only one cluster remains
                if clusters.max() == 0:
                    logging.info("No separable clusters")
                    return False

            if method == 'dendrogram':
                agg_file = os.path.join(config.paths.build, "data", subset + ".agg.loom")
                with loompy.connect(agg_file, 'r') as dsagg:
                    Z = dsagg.attrs.linkage
                    branch = cut_tree(Z, 2).T[0]
                clusters = np.array([branch[x] for x in ds.ca.Clusters])
                ds.ca.Split = clusters

            if method == 'cluster':
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
                f.write('  execution:\n')
                f.write(f'    n_cpus: {n_cpus}\n')
                f.write(f'    memory: {memory}\n')

    return True
