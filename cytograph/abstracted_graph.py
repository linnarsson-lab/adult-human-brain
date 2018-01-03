import numpy as np
import loompy
from scipy import stats, sparse
from statsmodels.sandbox.stats.multicomp import multipletests
from typing import *


# from numba import njit, jit, autojit
# @njit()
def _partition_edges_count(rows: np.ndarray, cols: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Counts the inter- and intra-partition edges on G

    Arguments
    ---------
    rows: np.ndarray(dtype=int)
        The row indexes of the nonzero entries of a Adjacency matrix.
        For esample in a scipy.coo_matrix this is the `row` attribute. From loompy is ds.get_edges("KNN", axis=1)[0]
    cols: np.ndarray(dtype=int)
        The column indexes of the nonzero entries of a Adjacency matrix.
        For esample in a scipy.coo_matrix this is the `col` attribute. From loompy is ds.get_edges("KNN", axis=1)[1]
    clusters: np.ndarray(dtype=int)
        The labels of the partition of the graph.

    Returns
    -------
    K: np.ndarray
        K[i,j] is the number of edgest from cluster i to cluster j, this should be asymmetric as the knn graph is

    Note
    ----
    The looped version of the function function could be numba jitted for a x200 speedup
    """
    n_clusters = int(np.max(clusters) + 1)
    K = np.zeros((n_clusters, n_clusters))
    N = len(rows)

    np.add.at(K, (clusters[rows], clusters[cols]), 1)
    # the one above is a not buffered operation: it is different from K[clusters[rows], clusters[cols]) += 1
    # and it is equivalent of the following:
    # for i in range(N):
    #     K[clusters[rows[i]], clusters[cols[i]]] += 1

    return K


def adjacency_confidence(knn: sparse.coo_matrix, clusters: np.ndarray, symmetric: bool=False) -> np.ndarray:
    """Return adjacency confidence for the abstracted graph Gx

    Arguments
    ---------
    knn: sparse.coo_matrix shape (cells, cells)
        sparse KNN matrix representation of G
    clusters: np.ndarray, int64[:]
        cluster indexes as they would be provided by the function np.unique(x, return_inverse=True)[1]
        (e.g. dtype int, no holes)
    symmetric: bool, default=True
        force symmetry by considering edges in both directions

    Returns
    -------
    np.ndarray, double[:, :]
        Confidence of the adjeciency matrix

    Notes
    -----
    Some notes from Wolf et al. 2017, Supl. Note 3 and 4

    - Edge statistics
    K[i,j] cells of cluster i that connect to cluster j, this should be asymmetric as the knn graph
    theta[i] it the frequency of an edge to connect with cluster i.
        theta = K.sum(0) / K.sum()
    Random model of randomly connected partitions:
        K / K.sum() = theta[:, None] * theta[None, :]  # expected value
        K / K.sum() ~ Bernoulli(theta[:, None] * theta[None, :])
    modularity ( estatistics that comparse the oberved freq with expected)
        M = (K / K.sum()) - (theta[:, None] * theta[None, :])
    sum of bernuolli -> binomial -> if N>20 is well aprox by Gaussian
    p = theta[:, None] * theta[None, :]
    n = K.sum()
    var[M] =  p (1 - p) / n
    """
    assert clusters.dtype == "int", "Argument clusters should have dtype == 'int'"
    K = _partition_edges_count(knn.row, knn.col, clusters)  # Inter and intra partition edges
    np.fill_diagonal(K, 0)
    if symmetric:
        K = K + K.T
    
    k = len(knn.row) / len(clusters)
    # if symmetric:
    #     k = k *2
    _, counts = np.unique(clusters)
    total_n = k * counts
    expected = total_n[:, None] * total_n[None, :] / np.sum(total_n)**2
    actual = K / np.sum(total_n)
    variance = expected * (1 - expected) / np.sum(total_n)

    confidence = np.zeros(K.shape, dtype="double")
    confidence[actual > expected] = 1
    confidence[actual <= expected] = 2 * stats.norm.cdf(actual[actual <= expected],
                                                        expected[actual <= expected],
                                                        np.sqrt(variance[actual <= expected]))
    confidence[actual < 1e-12] = 0
    confidence = confidence
    np.fill_diagonal(confidence, 0)

    return confidence


def velocity_summary(vlm: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize velocity field at the cluster level

    Arguments
    ---------
    vlm: velocyto.VelocytoLoom
        The main velocyto object

    Returns
    -------
    significant: np.ndarray
        Whether the transition probability between two clsuters is bigger than expeted by chance
    trans: np.ndarray
        Velocity derived transition probability matrix at the cluster level
    expected_tr: np.ndarray
        The expected transtion probability taking in consideration only then number of cells per cluster.

    """

    # Count of cells for each clusters
    _, counts = np.unique(vlm.cluster_ix, return_counts=True)
    
    try:
        sparse_TP = sparse.coo_matrix(vlm.transition_prob)  # NOTE: probably using nonzero/where is better
    except AttributeError:
        # Convert correlation coefficient to transition probability
        sigma_corr = 0.05
        vlm.transition_prob = np.exp(vlm.corrcoef / sigma_corr) * vlm.embedding_knn.A 
        vlm.transition_prob /= vlm.transition_prob.sum(1)[:, None]
        sparse_TP = sparse.coo_matrix(vlm.transition_prob)

    Z = np.zeros((len(counts), len(counts)))
    np.add.at(Z, (vlm.cluster_ix[sparse_TP.row], vlm.cluster_ix[sparse_TP.col]), sparse_TP.data)
    # Normalizations: dividing by counts[:, None]) it normalizes so that if all i connect to j then Z[i,j] ~= 1
    # However still clusters with more cells are more often acceptors od edges just by chance
    # this is taken in account later.
    Z_norm = Z / counts[:, None]  # Z_norm.sum(1) == np.ones
    np.fill_diagonal(Z_norm, 0)
    # consiter the netto flux, e.g. if i-> j 0.1 and j->i 0.04 change to i->j 0.06 and j->i 0
    delta_Z = Z_norm - Z_norm.T
    # Transition matrix without diagonal
    trans = np.triu(np.clip(delta_Z, 0, 1)) + np.tril(np.clip(delta_Z, 0, 1))

    expected_tr = counts / np.sum(counts)
    margin = np.minimum(np.min(expected_tr), (1 - expected_tr) / 3.)  # take care of corner case with few clusters
    thrs = (expected_tr + margin)[None, :]
    significant = trans > thrs

    return significant, trans, expected_tr


def plot_confidence_and_velocity(trans: np.ndarray, expected_tr: np.ndarray, confidence: np.ndarray, thrs_conf: float=0.05) -> None:
    import matplotlib.pyplot as plt
    plt.imshow(trans, cmap="Blues")
    yy, xx = np.where(trans > (expected_tr[None, :] + np.min(expected_tr)))
    plt.scatter(xx, yy, marker="+", c="orange", s=100)
    plt.plot(np.linspace(-0.5, trans.shape[1] - 0.5), np.linspace(-0.5, trans.shape[1] - 0.5), c="k")
    xx, yy = np.where(confidence > thrs_conf)
    plt.scatter(xx, yy, marker="o", s=100, facecolor="none", edgecolor="orange", lw=2)
    plt.grid("off")


class GraphAbstraction:
    def __init__(self, kind: str="simple") -> None:
        self.kind = kind

    def _compute_confidence(self, ds: loompy.LoomConnection) -> None:
        a, b, w = ds.get_edges("KNN", axis=1)  # consider using MKNN
        knn = sparse.coo_matrix((w, (a, b)), shape=(ds.shape[1], ds.shape[1]))
        clusters = np.unique(ds.col_attrs["Clusters"], return_inverse=True)[1]
        self.confidence = adjacency_confidence(knn, clusters)

    def abstract(self, ds: loompy.LoomConnection, thresh_confid: float=0.5, unidirectional: bool=False) -> sparse.coo_matrix:
        """Return a sparse matrix representation of the absracted graph

        Arguments
        ---------
        ds: LoomConnection
            Dataset
        thresh_confid: float
            Edges with conficedence below `thresh_confid` will be dropped
        unidirectional: bool:
            Whether to retrurn the upper triangle of the connectivity matrix

        Returns
        -------
        sparse.coo_matrix
        """
        self._compute_confidence(ds)
        confidence = np.copy(self.confidence)
        confidence[confidence < thresh_confid] = 0
        if unidirectional:
            confidence = np.triu(confidence)
        return sparse.coo_matrix(confidence)


# Incomplete code
def distance_based_confidence(knn: sparse.coo_matrix, clusters: np.ndarray, thrsh: float) -> np.ndarray:
    """
    - Random-walk based measure for connectivity

    rationale is that: d[i,j] ~ n_paths[i,j]
    d = pdist(X)  # d[i, j] = dist(X[i,:], X[j,:])
    where dist is tandom-walk based measure
    c1[cl1, cl2] = cdist(cl1, cl2, "random_walk").min() # measure of connectivity
    c3[cl1, cl2] = cdist(cl1, cl2, "random_walk").median() # alternative measure of connectivity
    heuristic
    c1[cl1, cl2] if c1_significant else c3[cl1, cl2]

    to compare different Tx of Gx one can compute
    
    q = ones(Tx.shape)
    c1T = c1.median()
    ratio = c1T / c1
    q[ratio >= 1] = exp(1 - ratio[ratio >= 1])

    q3 = ones(Tx.shape)
    c3T = c3.median()
    ratio = c3T / c3
    c3T = c3.median()
    ratio = c3T / c3
    q3[ratio >= 1] = exp(1 - ratio[ratio >= 1])

    q[q>thrsh] = q3[q>thrsh] # heuristic above


    (key funciton trace existing connections)
    """
    raise NotImplementedError("Finding the random walk distance is not implemented yet")
    unq_cls = np.unique(clusters)
    N = len(unq_cls)
    c1 = np.zeros((N, N))
    c3 = np.zeros((N, N))
    # for cl1 in unq_cls:
    #   for cl2 in unq_cls:
    #       ! dist = cdist(X[cluster == cl1, :], X[cluster == cl2, :], "random_walk")
    #       c1[cl1, cl2] = np.min(dist)
    #       c3[cl1, cl2] = np.median(dist)

    c1T = np.median(c1)
    ratio = c1T / c1
    confidence = np.zeros((N, N))
    confidence[ratio > 1] = np.exp(1 - ratio[ratio >= 1])

    # confidence fallback
    fallback = np.zeros((N, N))
    c3T = np.median(c3)
    ratio = c3T / c3
    fallback[ratio >= 1] = np.exp(1 - ratio[ratio >= 1])

    confidence[confidence < thrsh] = fallback[confidence < thrsh]
    
    np.fill_diagonal(confidence, 0)  # NOTE: not sure about this

    return confidence


def aga() -> None:
    """Using the single-cell graph G = (N,E) Generate a much simpler abstracted graph Gx, whose nodes
    correspond to groups and whose edge weights quantify the connectivity between groups.
    Following paths along nodes in Gx means following an ensemble of single-cell paths that pass through the corresponding groups in G

    In the visualization of abstracted graphs, edge width is proportional to the confidence in the presence of an actual connection
    (altrnatives: random walk, topological data analysis)

    Supplemental Note 2
    -------------------
    """
    raise NotImplementedError("Place-holder")


def graph_partitioning() -> None:
    """Partitioning could be done directly using Lovain
    splits_segments()
        detect_splits()
        
        postprocess_segments()
        
        set_segs_names()
        
        order_pseudotime()
    """
    raise NotImplementedError("Place-holder")


def tree_detection() -> None:
    """
    iterative_matching or minimum spanning tree

    """
    raise NotImplementedError("Place-holder")