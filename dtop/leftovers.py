def leftover_code():
    # Then, use HDBSCAN to identify gene clusters
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_regulon_size)
    self.regulon_labels = hdb.fit_predict(tsne)

def factorize_regulons(self):
    return np.array([self._factorize_regulon(r).mean(axis=1) for r in self.regulon_labels])

def _factorize_regulon(self, label):
    """
    Factorize the given regulon using bayesian non-negative matrix factorization

    Args:
        label (int):	The regulon to factorize

    """
    # TODO: truncate list of genes to max m genes to save on RAM
    genes = (self.regulons == label)
    if not genes.any():
        raise ValueError("Regulon doesn't exist")

    y = self.data[genes, :]
    c = y.shape[1]
    g = y.shape[0]
    data = {
        "C": c,
        "G": g,
        "y": y
    }
    stan = CmdStan()
    result = stan.fit("bnnmf", data, method="sample", debug=False)
    return result["alpha"]

def MkNN_cells(self, genes, k):
    """
    Compute a MkNN graph based on the given set of genes
    
    Args:
        genes (numpy array of bool):		The gene selection
        k (int):							Number of nearest neighbours to consider
    """

    if len(genes) == 0:
        raise ValueError("No genes were selected")
    
    annoy = AnnoyIndex(genes.sum(), metric = 'angular')
    for i in xrange(self.data.shape[1]):
        vector = self.data[genes, i]
        annoy.add(i, vector)
    
    annoy.build(10)

    # TODO: save the index, then use multiple cores to search it 

    # Compute kNN and distances for each cell, in sparse matrix IJV format
    d = self.data.shape[i]
    I = np.empty(d*k)
    J = np.empty(d*k)
    V = np.empty(d*k)
    for i in xrange(d):	
        (nn, w) = annoy.get_nns_by_item(i, k, include_distances = True)
        I[i*k:i*(k+1)] = [i]*k
        J[i*k:i*(k+1)] = nn
        V[i*k:i*(k+1)] = w

    kNN = sparse.coo_matrix((V,(I,J)),shape=(d,d))

    # Compute Mutual kNN
    kNN = kNN.tocsr()
    t = kNN.transpose(copy=True)
    self.MkNN = 1 - (kNN * t) # This removes all edges that are not reciprocal, and converts distances to similarities

def nested_blocks_on_genes(self, k):
    """
    Compute a MkNN graph based on the given set of genes
    
    Args:
        genes (numpy array of bool):		The gene selection
        k (int):							Number of nearest neighbours to consider
    """

    if len(genes) == 0:
        raise ValueError("No genes were selected")
    
    annoy = AnnoyIndex(genes.sum(), metric = 'angular')
    for i in xrange(self.data.shape[1]):
        vector = self.data[genes, i]
        annoy.add(i, vector)
    
    annoy.build(10)

    # TODO: save the index, then use multiple cores to search it 

    # Compute kNN and distances for each cell, in sparse matrix IJV format
    d = self.data.shape[i]
    I = np.empty(d*k)
    J = np.empty(d*k)
    V = np.empty(d*k)
    for i in xrange(d):	
        (nn, w) = annoy.get_nns_by_item(i, k, include_distances = True)
        I[i*k:i*(k+1)] = [i]*k
        J[i*k:i*(k+1)] = nn
        V[i*k:i*(k+1)] = w

    kNN = sparse.coo_matrix((V,(I,J)),shape=(d,d))

    # Compute Mutual kNN
    kNN = kNN.tocsr()
    t = kNN.transpose(copy=True)
    self.MkNN = 1 - (kNN * t) # This removes all edges that are not reciprocal, and converts distances to similarities


bnnmf = """
# Model of a single regulon

data {
    int <lower=0> G;                    # number of genes
    int <lower=0> C;                    # number of cells
    int <lower=0> y[G, C];              # observed molecule counts
}

parameters {
    vector <lower=0,upper=1> [C] alpha;         # coefficient for each cell
    row_vector <lower=1> [G] beta;      # coefficient for each gene
    real <lower=0> r;                   # overdispersion
}

model {
    row_vector [C] mu[G];
    real rsq;

    # Noise model
    r ~ cauchy(0,1);
    rsq <- square(r + 1) - 1;

    # Matrix factorization
     beta ~ cauchy(1, 5);
    alpha ~ beta(0.5,1.5); #cauchy(0, 5);

    for (g in 1:G) {
        # compute hidden expression level
        mu[g] <- (alpha * beta[g])';

        # observations are NB-distributed with noise
        y[g] ~ neg_binomial(mu[g] / rsq, 1 / rsq);
    }
}
"""
