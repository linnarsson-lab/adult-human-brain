# -*- coding: utf-8 -*-
import loompy
import scipy
import scipy.misc
import scipy.ndimage
from scipy.optimize import minimize
from scipy import sparse
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import affinity_propagation
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import numpy as np
import hdbscan
from annoy import AnnoyIndex
import logging



def knn_similarities(ds, k=50, kth=5, r=10, annoy_trees=50, cells=None, n_genes=1000, n_components=200, epsilon=0.001, mutual=True, metric='euclidean'):
	"""
	Compute knn similarity matrix for the given cells

	Args:
		ds (LoomConnecton):		Loom file
		k (int):				Number of nearest neighbours to include in graph
		kth (int):				Nearest neighbour to use for local kernel width
		r (int):				Number of remote neighbours per pair of components	
		annoy_trees (int):		Number of Annoy trees used for kNN approximation
		cells (int array):		Indices of cells to include in the graph
		epsilon (float):		Minimum similarity for long-range relations
		mutual (bool): 			If true, retain only mutual nearest neighbors

	Returns:
		knn (sparse matrix):	Matrix of similarities of k nearest neighbors 
		sigma (numpy array):	Nearest neighbor similarity for each cell
		genes (array of int):	Selection of genes that was used for the graph
	"""
	if cells is None:
		cells = np.array(range(ds.shape[1]))

	# Compute an initial gene set
	ds.feature_selection(n_genes, method="SVR")
	genes = np.where(ds.row_attrs["_Excluded"]==0)[0]

	# Pick a random set of (up to) 1000 cells
	cells_sample = cells
	if len(cells_sample) > 1000:
		cells_sample = np.random.choice(cells_sample, size=1000, replace=False)

	# Perform PCA based on the gene selection and the cell sample
	pca = PCA(n_components=n_components)
	pca.fit(np.log(ds[:,cells][genes,:]+1).transpose())

	annoy = AnnoyIndex(n_components, metric = metric)
	for ix in cells:
		vector = ds[:,ix][genes]
		transformed = pca.transform(np.log(ds[:,ix][genes,np.newaxis]+1).transpose())[0]
		annoy.add_item(ix, transformed)
	
	annoy.build(annoy_trees)

	# Compute kNN and similarities for each cell, in sparse matrix IJV format
	d = len(cells)
	I = np.empty(d*k)
	J = np.empty(d*k)
	V = np.empty(d*k)
	sigma = np.empty((d,)) # The local kernel width  
	for i in xrange(d):	
		(nn, w) = annoy.get_nns_by_item(i, k, include_distances = True)
		w = np.array(w)
		I[i*k:(i+1)*k] = [i]*k
		J[i*k:(i+1)*k] = nn
		V[i*k:(i+1)*k] =  w	
		sigma[i] = w[kth-1]

	# k nearest neighbours
	kNN = sparse.coo_matrix((V,(I,J)),shape=(d,d))

	# Create remote links
	(n_connected, labels) = sparse.csgraph.connected_components(kNN, directed='False')
	print "n_connected: ", n_components
	if n_connected > 1:
		n = (n_connected*n_connected/2 - 1)*r
		I = np.empty(n*r)
		J = np.empty(n*r)
		V = np.empty(n*r)
		for i in xrange(n_connected):
			for j in xrange(n_connected):
				if i >= j:
					continue
				# Generate r random links between this pair of clusters
				c1 = np.random.choice(labels.where(labels==i), r)
				c2 = np.random.choice(labels.where(labels==j), r)

				# TODO: Remove duplicates (currently, they will be summed)

				w = np.empty_like(c1)
				for i,(x,y) in enumerate(zip(c1,c2)):
					w[i] = annoy.distance(x,y)

				w = np.array(w)

				# Put them in the array
				I[i*n_connected + j : i*n_connected + j + r] = c1
				J[i*n_connected + j : i*n_connected + j + r] = c2
				V[i*n_connected + j : i*n_connected + j + r] = w

	# r remote neighbours
	V = np.minimum(V + epsilon, V)	# Add a minimum long-range similarity
	
	rRN = sparse.coo_matrix((V,(I,J)),shape=(d,d))
	
	# Merge the two sparse matrices
	# See http://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
	data = scipy.concatenate((kNN.data,rRN.data))
	rows = scipy.concatenate((kNN.row, rRN.row))
	cols = scipy.concatenate((kNN.col, rRN.row)) 

	# Convert to similarities by rescaling and subtracting from 1
	data = data / data.max()
	data = 1 - data
	sigma = sigma / sigma.max()
	sigma = 1 - sigma

	kNN = sparse.coo_matrix((data,(rows,cols)), shape=(d,d))

	if mutual:
		# Compute Mutual kNN
		kNN = kNN.minimum(kNN.transpose()) # This removes all edges that are not reciprocal
	else:
		# Make all edges reciprocal
		kNN = kNN.maximum(kNN.transpose()) # This duplicates all edges that are not reciprocal
	kNN = kNN.tocoo() # Go back to COO format
			
	return (kNN, sigma, genes, pca)


def sparse_dmap(m, sigma):
	"""
	Compute the diffusion map of a sparse similarity matrix

	Args:
		m (sparse.coo_matrix):	Sparse matrix of similarities in [0,1]
		sigma (numpy.array):    Array of nearest neighbor similarities

	Returns:
		tsym (sparse.coo_matrix):  Symmetric transition matrix T

	Note:
		The input is a matrix of *similarities*, not distances, which helps sparsity because very distant cells
		will have similarity near zero (whereas distances would be large).

	"""
	# Convert sigma to distances
	sigma = 1 - sigma

	# The code below closely follows that of the diffusion pseudotime paper (supplementary methods) 

	# sigma_x^2 + sigma_y^2 (only for the non-zero rows and columns)
	ss = np.power(sigma[m.row],2) + np.power(sigma[m.col],2)

	# In the line below, '1 - m.data' converts the input similarities to distances, but only for the non-zero entries
	# That's fine because in the end (tsym) the zero entries will end up zero anyway, so no need to involve them
	kxy = sparse.coo_matrix((np.sqrt(2*sigma[m.row]*sigma[m.col]/ss) * np.exp(-np.power(1-m.data,2)/(2*ss)),(m.row,m.col)),shape=m.shape)
	zx = kxy.sum(axis=1).A1
	wxy = sparse.coo_matrix((kxy.data/(zx[kxy.row]*zx[kxy.col]),(kxy.row, kxy.col)), shape=kxy.shape)
	zhat = wxy.sum(axis=1).A1
	tsym = sparse.coo_matrix( (wxy.data * np.power(zhat[wxy.row], -0.5) * np.power(zhat[wxy.col], -0.5),(wxy.row, wxy.col)), shape=wxy.shape)
	return tsym

def dpt(f, t, k, path_integral=True):
	"""
	Compute k steps of pseudotime evolution from starting vector f and transition matrix t

	Args:
		f (numpy 1D array):			The starting vector of N elements
		t (numpy sparse 2d array):	The right-stochastic transition matrix
		k (int):					Number of steps to evolve the input vector
		path_integral (bool):		If true, calculate all-paths integral; otherwise calculate time evolution 
	
	Note: 
		This algorithm is basically the same as Google PageRank, except we start typically from 
		a single cell (or a few cells), rather than a uniform distribution, and we don't make random jumps.
		See: http://michaelnielsen.org/blog/using-your-laptop-to-compute-pagerank-for-millions-of-webpages/ and 
		the original PageRank paper: http://infolab.stanford.edu/~backrub/google.html
	"""
	f = sparse.csr_matrix(f)
	result = np.zeros(f.shape)
	if path_integral:
		for ix in xrange(k):
			f = f.dot(t)
			result = result + f
		return result.A1
	else:
		for ix in xrange(k):
			f = f.dot(t)
		return f.A1

class MapperNode(object):
	def __init__(self, ds, persistence, cells, pseudotime):
		self.ds = ds
		self.persistence = persistence
		self.cells = cells
		self.n = len(cells)
		self.pseudotime = pseudotime

def mapper(ds, genes, cells, f, step, overlap=1.5):
	"""
	Construct the topological skeleton graph of the selected cells

	Args:
		cells (int array): 	The cell selection
		f (float array):	The pseudotime for each cell
		step (float):		Pseudotime step size
		overlap (float):	The overlap factor (>1.0, e.g. 1.5)
	"""
	if cells is None:
		cells = np.fromiter(xrange(ds.shape[1]),dtype="int")

	f0 = f.min()
	f1 = f.max()
	bin_size = step*overlap
	ix = f0
	nodes = []
	while ix < f1:
		# Select the cells that fall in the bin based on their pseudotime
		selection = cells[np.logical_and(f > ix, f < ix + bin_size)]
		# If there were more than two cells, perform clustering
		if selection.shape[0] > 2:
			# Cluster them to separate distant manifold branches
			data = np.log(ds[:,selection][genes,:]+1)
			dists = pairwise_distances(data.transpose(), metric="cosine")
			clusterer = hdbscan.HDBSCAN(metric='precomputed', allow_single_cluster=True)
			clusterer.fit(dists.astype("float64"))
			labels = clusterer.labels_
			n_clusters = labels.max() + 1
			print n_clusters
			# Create skeleton graph nodes and assign cells to them
			for n in xrange(n_clusters):
				nodes.append(MapperNode(ds, clusterer.cluster_persistence_, selection[labels == n], ix + bin_size/2))
		elif selection.shape[0] > 0:
			nodes.append(MapperNode(ds, 0, selection, ix + bin_size/2))
		ix += step

	graph = sparse.dok_matrix((len(nodes),len(nodes)))
	for i,n1 in enumerate(nodes):
		for j,n2 in enumerate(nodes):
			if j >= i:
				continue
			shared = np.intersect1d(n1.cells, n2.cells)
			if len(shared) > 0:
				graph[i,j] = len(shared)

	return graph, nodes

def cluster(ds, genes, cells):
	"""
	Cluster the given cells with respect to the given genes

	Args:

		ds (LoomConnection): 	The loom data file
		genes (array of int):	The selected genes
		cells (array of int):	The selected cells
	
	Returns:
		prototypes (array of int):	Indexes of the prototype cell for each cluster
		labels (array of int):		Cluster labels
		n_clusters (int):			Number of clusters
	"""
	data = ds[:,cells][genes,:]
	if data.shape[0] == 0 or data.shape[1] == 0:
		return (None,None,0)
	dists = pairwise_distances(data.transpose(), metric="cosine")

	prototypes, labels = affinity_propagation(dists)
	return (prototypes, labels, np.max(labels) + 1)
