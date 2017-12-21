import numpy as np
from scipy import stats, sparse


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
    # the one above is a not buffered oberation: it is different from K[clusters[rows], clusters[cols]) += 1
    # and it is equivalent of the following:
    # for i in range(N):
    #     K[clusters[rows[i]], clusters[cols[i]]] += 1

    return K


def adjacency_confidence(knn: sparse.coo_matrix, clusters: np.ndarray) -> np.ndarray:
    """Retrun adjacency confidence for the abstracted graph Gx confidenge

    Arguments
    ---------
    knn: sparse.coo_matrix shape (cells, cells)
        sparse KNN matrix representation of Gx
    clusters: np.ndarray, int64[:]
        cluster indexes as they would be provided by the function np.unique(x, return_inverse=True)[1]
        (e.g. dtype int, no holes)

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
    n_edges = K.sum()
    theta = K.sum(0) / n_edges
    p = theta[:, None] * theta[None, :]  # Expected frequency
    q = K / n_edges  # Actual frequency
    M = q - p
    sigma = np.sqrt(p * (1 - p) / n_edges)  # variance from linear combination of n bernoulli variables

    confidence = np.zeros(M.shape, dtype="double")
    confidence[M > 0] = 1  # Not sure about this
    confidence[q < 1e-12] = 0
    confidence[M <= 0] = 2 * stats.norm.cdf(M, 0, sigma)

    np.fill_diagonal(confidence, 0)  # NOTE: not sure about this

    return confidence


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