from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def biPCA(data, cells, n_splits=10, n_components=20):
	# data is numpy matrix with genes already selected
	# Perform PCA
	selection = cells
	if selection.shape[0] > 10000:
		selection = np.random.choice(selection, 10000, replace=False)
	pca = PCA(n_components=n_components)
	tranform = pca.fit(data[:, selection].transpose())
	data_t = pca.transform(data[:, cells].transpose())
	# TODO: select significant components

	# Do k-means best of five
	best_labels = None
	best_score = -1
	for _ in range(5):
		labels = KMeans(n_clusters=2).fit_predict(data_t)
		if best_labels == None:
			best_labels = labels
		else:
			score = silhouette_score(data_t, cluster_labels)
			if score > best_score:
				best_score = score
				best_labels = labels

	cells_left = cells[best_labels == 0]
	cells_right = cells[best_labels == 1]
