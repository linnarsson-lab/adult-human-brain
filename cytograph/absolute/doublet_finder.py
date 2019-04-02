# This function is based on doubletFinder.R as forwarded by the Allen Institute:
#
    # "Doublet detection in single-cell RNA sequencing data
    #
    # This function generaetes artificial nearest neighbors from existing single-cell RNA
    # sequencing data. First, real and artificial data are merged. Second, dimension reduction
    # is performed on the merged real-artificial dataset using PCA. Third, the proportion of
    # artificial nearest neighbors is defined for each real cell. Finally, real cells are rank-
    # ordered and predicted doublets are defined via thresholding based on the expected number
    # of doublets.
    #
    # @param seu A fully-processed Seurat object (i.e. after normalization, variable gene definition,
    # scaling, PCA, and tSNE).
    # @param expected.doublets The number of doublets expected to be present in the original data.
    # This value can best be estimated from cell loading densities into the 10X/Drop-Seq device.
    # @param porportion.artificial The proportion (from 0-1) of the merged real-artificial dataset
    # that is artificial. In other words, this argument defines the total number of artificial doublets.
    # Default is set to 25%, based on optimization on PBMCs (see McGinnis, Murrow and Gartner 2018, BioRxiv).
    # @param proportion.NN The proportion (from 0-1) of the merged real-artificial dataset used to define
    # each cell's neighborhood in PC space. Default set to 1%, based on optimization on PBMCs (see McGinnis,
    # Murrow and Gartner 2018, BioRxiv).
    # @return An updated Seurat object with metadata for pANN values and doublet predictions.
    # @export
    # @examples
    # seu <- doubletFinder(seu, expected.doublets = 1000, proportion.artificial = 0.25, proportion.NN = 0.01)"


import loompy
import numpy as np
import cytograph as cg
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent
from scipy import sparse

def doublet_finder(ds: loompy.LoomConnection, use_pca: bool = False, proportion_artificial: float = 0.20,
                  k: int = None):

    # Step 1: Generate artificial doublets from Seurat object input
    print("Creating artificial doublets")
    n_real_cells = ds.shape[1]
    n_doublets = int(n_real_cells / (1 - proportion_artificial) - n_real_cells)
    doublets = np.zeros((ds.shape[0], n_doublets))
    for i in range(n_doublets):
        a = np.random.choice(ds.shape[1])
        b = np.random.choice(ds.shape[1])
        doublets[:, i] = ds[:, a] + ds[:, b]

    data_wdoublets = np.concatenate((ds[:, :], doublets), axis=1)

    print("Feature selection and dimensionality reduction")
    normalizer = cg.Normalizer(False)
    normalizer.fit(ds)
    genes = cg.FeatureSelection(2000).fit(ds, mu=normalizer.mu, sd=normalizer.sd)
    if use_pca:
        # R function uses log2 counts/million
        f = np.divide(data_wdoublets.sum(axis=0), 10^6)
        norm_data = np.divide(data_wdoublets, f)
        norm_data = np.log(norm_data + 1)
        pca = PCA(n_components=50).fit_transform(norm_data[genes, :].T)
    else:
        data = sparse.coo_matrix(data_wdoublets[genes, :]).T
        hpf = cg.HPF(k=64, validation_fraction=0.05, min_iter=10, max_iter=200, compute_X_ppv=False)
        hpf.fit(data)
        theta = (hpf.theta.T / hpf.theta.sum(axis=1)).T
    
    if k is None:
        k = int(np.min([100, ds.shape[1] * 0.01]))

    print("Initialize NN structure with k =", k)
    if use_pca:
        knn_result = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=4)
        knn_result.fit(pca)
        knn_dist, knn_idx = knn_result.kneighbors(X=pca, return_distance=True)

        num = ds.shape[1]
        knn_result1 = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=4)
        knn_result1.fit(pca[0:num, :])
        knn_dist1, knn_idx1 = knn_result1.kneighbors(X=pca[num + 1:, :], n_neighbors=10)
    else:
        knn_result = NNDescent(data=theta, metric=cg.metrics.jensen_shannon_distance)
        knn_idx, knn_dist = knn_result.query(theta, k=k)

        num = ds.shape[1]
        knn_result1 = NNDescent(data=theta[0:num, :], metric=cg.metrics.jensen_shannon_distance)
        knn_idx1, knn_dist1 = knn_result1.query(theta[num + 1:, :], k=k)

    dist_th = np.mean(knn_dist1.flatten()) + 1.64 * np.std(knn_dist1.flatten())

    doublet_freq = np.logical_and(knn_idx > ds.shape[1], knn_dist < dist_th)
    doublet_freq = doublet_freq[0:ds.shape[1], :]

    mean1 = doublet_freq.mean(axis=1)
    mean2 = doublet_freq[:, 0:int(np.ceil(k/2))].mean(axis=1)
    doublet_score = np.maximum(mean1, mean2)
    return doublet_score
