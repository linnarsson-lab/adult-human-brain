import os
import logging
import loompy
import numpy as np
import numpy_groupies as npg
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
from ..plotting.colors import colorize
from ..species import Species
from ..enrichment import FeatureSelectionByEnrichment


def merge_subset(subset: str, config, threshold: int = 5) -> None:

    loom_file = os.path.join(config.paths.build, "data", subset + ".loom")

    with loompy.connect(loom_file) as ds:

        # Copy original clusters
        ds.ca.ClustersPremerge = np.copy(ds.ca.Clusters)
        logging.info(f'Starting with {ds.shape[1]} cells in {ds.ca.Clusters.max() + 1} clusters')
        logging.info(f'Threshold: {threshold} enriched genes with qval < 0.001, enrichment > 5')

        merge_flag = True

        while merge_flag:
            # Renumber clusters
            clusters = ds.ca.Clusters

            # Stop if only one cluster
            n = clusters.max()
            if n == 0:
                logging.info('Only one cluster found.')
                merge_flag = False
                break

            # Calculate enrichment statistics
            features = FeatureSelectionByEnrichment(1, Species.mask(ds, config.params.mask), findq=True)
            features.fit(ds)
            # Count statistically enriched genes for each cluster
            scores = np.count_nonzero((features.qvals < 0.001) & (features.enrichment > 5), axis=0)
            scores = np.array(scores)
            logging.info(scores)

            # If more than one cluster has fewer enriched genes than threshold
            if (scores < threshold).sum() > 1:
                # Calculate cluster distances on the TSNE
                mu = npg.aggregate(clusters, ds.ca.TSNE.T, func=lambda x: np.median(x), axis=1, fill_value=0)
                D = squareform(pdist(mu.T))
                np.fill_diagonal(D, np.inf)
                # Find cluster with the minimum score
                c1 = np.argmin(scores)
                # Find the nearest cluster
                c2 = np.argmin(D[c1])
                # Merge
                logging.info(f'Merging clusters {c1} and {c2} into cluster {n + 1}')
                clusters[(clusters == c1) | (clusters == c2)] = n + 1
            else:
                merge_flag = False

            _, ds.ca.Clusters = np.unique(clusters, return_inverse=True)

        # Plot unmerged and merged clusters
        exportdir = os.path.join(config.paths.build, "merge", "plots")
        if not os.path.exists(exportdir):
            os.mkdir(exportdir)
        plt.figure(None, (10, 5))
        plt.subplot(121)
        plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.ClustersPremerge), s=5)
        plt.title('Pre-merge clusters')
        plt.axis('off')
        plt.subplot(122)
        plt.scatter(ds.ca.TSNE[:, 0], ds.ca.TSNE[:, 1], c=colorize(ds.ca.Clusters), s=5)
        plt.title('Post-merge clusters')
        plt.axis('off')
        plt.savefig(os.path.join(exportdir, f'{subset}.png'), dpi=150)
        plt.close()
