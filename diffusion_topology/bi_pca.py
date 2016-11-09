from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.special import polygamma
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.spatial.distance import cdist
import numpy as np
import logging

def broken_stick(n, k):
	"""
	Return a vector of the k largest expected broken stick fragments, out of n total

	Remarks:
		The formula uses polygamma to exactly compute (1/n)sum{j=k to n}(1/j) for each k

	Note:	
		According to Cangelosi R. BiologyDirect 2017, this method might underestimate the dimensionality of the data
		In the paper a corrected method is proposed
	
	"""
	return np.array( [((polygamma(0,1+n)-polygamma(0,x+1))/n) for x in range(k)] )



def kmeans(X, k, metric="correlation", n_iter=10):
    """Kmeans implementation that allows using correlation

    Arguments
    ---------
    X: np.array(float)   shape=(samples, features)
        Input data
    k: int
        Number of clusters
    metric: str or callable
        Can be:
        - "corralation"
        - "euclidean"
        - In principle it also supports whatever can be passed as metric argument of sklern.pairwise.paired_distances
        (e.g.To use cosine one needs to normalize similarly to correlation)

    Returns
    -------
    labels: np.array(int) shape=(samples,)
        labels of the clusters

    Notes
    -----

    - The code was implemented by reading Wikipedia so it is not super-professional, it follows Llyoid alrgorythm.
    - The normalization to do in the case of correlation distance was taken from MATLAB source.
    - The algorythm seems very fast but I am not sure I am taking in account all the corner cases
    - Sometimes in the E step a empty cluster migh form, this will make impossible the M step for one centroid,
      this was solved by relocating this centroid close to the point that is farthest from all the other centroids
    - An implementation that does not load data into memory is possible with few changes
    """
    if metric == "correlation":
        X = X - X.mean(1)[:,None]
        X = X / np.sqrt( sum(X**2,1) ) # divide by unnormalized standard deviation over axis=0
        from scipy.stats.stats import pearsonr
        corr_dist = lambda a,b: 1 - pearsonr(a,b)[0]
        metric_f = corr_dist
    else:
        metric_f = metric
        
    # Start from infinite inertia
    best_inertia = np.inf
    # And run the algorythm n_iter times keeping track of the one with smallest intertai
    for _ in range(n_iter):
        # Initialize labels and tolerance
        final_label = -np.ones(X.shape[0])
        tol = 1e-16
        # Randomly choose centroids from the dataset
        ix = np.random.choice(X.shape[0],k,replace=False) 
        updated_centroids = X[ix,:].copy()

        # Perform EM
        while True:
            # Expectation Step - assign cell to closest centroid
            centroids = updated_centroids.copy()
            D = cdist(X, centroids, metric=metric)
            label = np.argmin(D,1)

            # Maximization step - relocate centroid to the average of the clusters
            for i in range(k):

                query = (label == i)
                if sum(query): # The cluster is not empty
                    updated_centroids[i,:] = np.mean(X[query,:],0)
                else:
                    # Relocate the centroid to the sample that is further from al the other centroids
                    updated_centroids[i,:] = X[argmax( np.min(D,1) ), :]

                if metric == "correlation":
                    # This bit is taken from MATLAB source code.
                    # The rationale is that the centroids should be recentered 
                    updated_centroids = updated_centroids - updated_centroids.mean(1)[:,None]

            # If all the centroids are not uppdated (within a max tolerance) Stop updating
            if np.all(paired_distances(centroids, updated_centroids, metric=metric_f) < tol, 0):
                break
        # Calculate inertia and keep track of the iteration with smallest inertia
        inertia = sum( D[np.arange(X.shape[0]), label] )
        if inertia < best_inertia:
            final_label = label.copy()
    return final_label

def biPCA(data, n_splits=10, n_components=20, cell_limit=10000, smallest_cluster = 5, verbose=2):
	'''biPCA algorythm for clustering using PCA splits

	Args
	----
		data : np.matrix
			data is numpy matrix with genes already selected
		n_splits : int
			number of splits to be attempted
		n_components: int
			max number of principal components
		cell_limit : int
			the max number of cells that are used to calculate PCA, if #cells>cell_limit, 
			a random sample of cell_limit cells will be drawn
		level_of_verbosity: int
			0 for only ERRORS
			1 for WARNINGS and ERRORS
			2 for INFOS, WARNINGS and ERRORS
			3 for DEBUGGING, NFOS, WARNINGS and ERRORS

	Returns
	-------

	'''

	logging.basicConfig(format='%(message)s', level=[logging.ERROR,logging.WARNING,logging.INFO,logging.DEBUG][verbose])

	n_genes, n_cells = data.shape
	cell_labels_by_depth = np.zeros((n_splits+1, n_cells))
	
	# Run a iteration per level of depth
	for i_split in range(n_splits):
		logging.info( "Depth: %i" % i_split )
		running_label_id = 0
		parent_clusters = np.unique(cell_labels_by_depth[i_split, :])
		# Consider every parent cluster and split them one by one
		for parent in parent_clusters:
			current_cells_ixs = np.where( cell_labels_by_depth[i_split, :] == parent )[0]
			data_tmp = np.log2( data[:,current_cells_ixs] + 1)
			data_tmp -= data_tmp.mean(1)[:,None]

			# Perform PCA
			if current_cells_ixs.shape[0] > cell_limit:
				selection = np.random.choice(np.arange(current_cells_ixs.shape[0]), cell_limit, replace=False)
			else:
				selection = np.arange(current_cells_ixs.shape[0])
			pca = PCA(n_components=n_components)
			pca.fit( data_tmp[:,selection].T )
			data_tmp = pca.transform( data_tmp.T ).T

			# Select significant components using broken-stick model
			bs = broken_stick(n_genes, min(n_components, len(current_cells_ixs) ))
			sig = pca.explained_variance_ratio_ > bs
			
			# No principal component is significant, don't split
			if not np.any(sig):
				
				logging.debug( "No principal component is significant, don't split!" )
				cell_labels_by_depth[i+1, current_cells_ixs] = running_label_id
				running_label_id += 1
				continue
			logging.debug('%i principal components are significant' % sum(sig))

			if sum(sig) < n_components:
				# Take only the significant components
				first_non_sign = np.where(np.logical_not(sig))[0][0]
				data_tmp = data_tmp[:first_non_sign, : ]
			else:
				# take all the components
				first_non_sign = n_components

			# Perform KMEANS clustering
			# NOTE
			# by default scikit learn runs n_init iterations with different centroid initialization and
			# uses inertia as a criterion to choose the best solution (avoiding local minima)
			# `inertia` is defined as  the sum of squared distances to he closest centroid for all samples.
			# Silhouette score is the average of (b - a) / max(a, b), calcualted for each sample
			# (a) is mean intra-cluster distance  and (b) is the mean nearest-cluster distance  for each sample

			best_labels = np.zeros(data_tmp.shape[1])
			best_score = -1
			# TODO This could be parallelized 
			for _ in range(3):
				# Here we could use MiniBatchKMeans when n_cells > 10k
				labels = KMeans(n_clusters=2, n_init=3, n_jobs=1).fit_predict(data_tmp.T)
				
				# The simplest way to calculate silhouette is  score = silhouette_score(X, labels)
				# However a cluster size resilient compuataion is:
				scores_percell = silhouette_samples(data_tmp.T, labels)
				score = min( np.mean(scores_percell[labels==0]), np.mean(scores_percell[labels==1]) )
				if score > best_score:
					best_score = score
					best_labels = labels
			logging.debug("Proposed split (%i,%i) has best_score: %s" % (sum(best_labels==0),sum(best_labels==1), best_score) )
			ids, cluster_counts = np.unique(best_labels, return_counts=True)			
			
			# Check that no small cluster gets generated
			if min(cluster_counts) < smallest_cluster:
				# Reject split immediatelly and continue
				logging.debug( "A small cluster get generated, don't split'" )
				cell_labels_by_depth[i_split+1, current_cells_ixs] = running_label_id
				running_label_id += 1
				continue
			
			# Conside only the 500 genes with top loadings
			sum_loadings_per_gene =  np.abs(  pca.components_.T[:,:first_non_sign].sum(1) )
			topload_gene_ixs = np.argsort(sum_loadings_per_gene)[::-1][:500]

			# Here I use mannwhitney U instead of binomial test
			p_values = np.array([ test_gene(data, current_cells_ixs, i, best_labels, 0) for i in topload_gene_ixs])
			p_values = np.minimum(p_values, 1-p_values)
			rejected_null, q_values = multipletests(p_values, 0.05, 'fdr_bh')[:2]

			# Decide if we should accept the split
			# Here instead of arbitrary threshold on shilouette I could boostrap
			if (np.sum(rejected_null) > 5) and (best_score>0.01):
				# Accept
				logging.debug("Spltting with significant genes: %s; silhouette-score: %s" % (np.sum(rejected_null), best_score))
				cell_labels_by_depth[i_split+1, current_cells_ixs[best_labels == 0]] = running_label_id 
				cell_labels_by_depth[i_split+1, current_cells_ixs[best_labels == 1]] = running_label_id + 1
				running_label_id += 2
			else:
				# Reject
				logging.debug( "Don't split. Significant genes: %s (min=5); silhouette-score: %s(min=0.01)" % (np.sum(rejected_null), best_score) )
				cell_labels_by_depth[i_split+1, current_cells_ixs] = running_label_id
				running_label_id += 1
	return cell_labels_by_depth

def test_gene(ds, cells, gene_ix, label, group):
    b = np.log2(ds[gene_ix,:][cells]+1)[label != group]
    a = np.log2(ds[gene_ix,:][cells]+1)[label == group]
    (_,pval) = mannwhitneyu(a,b,alternative="greater")
    return pval