from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.special import polygamma
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
import numpy as np

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

def biPCA(data, n_splits=10, n_components=20, cell_limit=10000, smallest_cluster = 5):
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

	Returns
	-------

	'''

	n_genes, n_cells = data.shape
	cell_labels_by_depth = np.zeros((n_splits+1, n_cells))
	
	# Run a iteration per level of depth
	for i_split in range(n_splits):
		print "Depth: %i" % i_split
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
			bs = broken_stick(n_genes, n_components)
			sig = pca.explained_variance_ratio_ > bs
			
			if not np.any(sig):
				# No principal component is significant, don't try to split
				print "No principal component is significant, don't try to split"
				cell_labels_by_depth[i+1, current_cells_ixs] = running_label_id
				running_label_id += 1
				continue
			
			if sum(sig) < n_components:
				# Take only the significant components
				first_non_sign = np.where(np.logical_not(sig))[0][0]
				data_tmp = data_tmp[:, :first_non_sign]
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

			best_labels = None
			best_score = -1
			# TODO This could be parallelized 
			for _ in range(3):
				# Here we could use MiniBatchKMeans when n_cells > 10k
				labels = KMeans(n_clusters=2, n_init=3, n_jobs=1).fit_predict(data_tmp.T)
				if best_labels == None:
					best_labels = labels
				else:
					# The simplest way to calculate silhouette is  score = silhouette_score(X, labels)
					# However a cluster size resilient compuataion is:
					scores_percell = silhouette_samples(data_tmp.T, labels)
					score = min( np.mean(scores_percell[labels==0]), np.mean(scores_percell[labels==1]) )
					if score > best_score:
						best_score = score
						best_labels = labels
			print "Proposed split (%i,%i) has best_score: %s" % (sum(best_labels==0),sum(best_labels==1), best_score)
			ids, cluster_counts = np.unique(best_labels, return_counts=True)			
			
			# Check that no small cluster gets generated
			if min(cluster_counts) < smallest_cluster:
				# Reject split immediatelly and continue
				print ids, cluster_counts
				print "A small cluster get generated, don't split'"
				cell_labels_by_depth[i_split+1, current_cells_ixs] = running_label_id
				running_label_id += 1
				continue

			sum_loadings_per_gene =  np.abs(  pca.components_.T[:,:first_non_sign].sum(1) )
			topload_gene_ixs = np.argsort(sum_loadings_per_gene)[::-1][:500] # max top 500 genes are used 

			# Here I use mannwhitney U instead of binomial test
			p_values = np.array([ test_gene(data, current_cells_ixs, i, best_labels, 0) for i in topload_gene_ixs])
			p_values = np.minimum(p_values, 1-p_values)
			rejected_null, q_values = multipletests(p_values, 0.05, 'fdr_bh')[:2]
			# Decide if we should accept the split
			if (np.sum(rejected_null) > 0) and (best_score>0.01):
				# Accept
				cell_labels_by_depth[i_split+1, current_cells_ixs[best_labels == 0]] = running_label_id 
				cell_labels_by_depth[i_split+1, current_cells_ixs[best_labels == 1]] = running_label_id + 1
				running_label_id += 2
			else:
				# Reject
				print "Don't split becouse' becouse chekc parameters are: %s, %s" % (np.sum(rejected_null), best_score)
				cell_labels_by_depth[i_split+1, current_cells_ixs] = running_label_id
				running_label_id += 1
	return cell_labels_by_depth

def test_gene(ds, cells, gene_ix, label, group):
    b = np.log2(ds[gene_ix,:][cells]+1)[label != group]
    a = np.log2(ds[gene_ix,:][cells]+1)[label == group]
    (_,pval) = mannwhitneyu(a,b,alternative="greater")
    return pval