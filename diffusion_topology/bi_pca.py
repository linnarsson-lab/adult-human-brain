from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.special import polygamma

def broken_stick(n, k):
	"""
	Return a vector of the k largest expected broken stick fragments, out of n total

	Remarks:
		The formula uses polygamma to exactly compute (1/n)sum{j=k to n}(1/j) for each k
	"""
	return [((polygamma(0,1+n)-polygamma(0,x+1))/n) for x in range(k)]

def biPCA(data, cells, n_splits=10, n_components=20):
	# data is numpy matrix with genes already selected
	# Perform PCA
	selection = cells
	if selection.shape[0] > 10000:
		selection = np.random.choice(selection, 10000, replace=False)
	pca = PCA(n_components=n_components)
	pca.fit(data[:,selection].transpose())
	data_t = pca.transform(data[:,cells].transpose())

	# Select significant components using broken-stick model
	bs = broken_stick(n, n_components)
	sig = pca.explained_variance_ratio_ > bs
	if not np.any(sig):
		return (1, cells)
	data_t = data_t[:, sig]

	# Do k-means best of five
	best_labels = None
	best_score = -1
	for _ in range(5):
		labels = KMeans(n_clusters=2).fit_predict(data_t)
		if best_labels == None:
			best_labels = labels
		else:
			score = silhouette_score(X, cluster_labels)
			if score > best_score:
				best_score = score
				best_labels = labels

	cells_left = cells[best_labels == 0]
	cells_right = cells[best_labels == 1]


def test_gene(ds, cells, gene, partition, group):
    b = np.log(ds[np.where(ds.Accession == gene),:][0,cells]+1)[partition != group]
    a = np.log(ds[np.where(ds.Accession == gene),:][0,cells]+1)[partition == group]
    (_,pval) = mannwhitneyu(a,b,alternative="greater")
    return pval