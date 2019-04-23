import logging
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
import time

# USAGE
#
# scrub = Scrublet(counts_matrix)
# doublet_scores, predicted_doublets = scrub.scrub_doublets()
#

# All code here originally by Samuel L Wolock, Romain Lopez, and Allon M Klein
# See: https://github.com/AllonKleinLab/scrublet
# See: https://www.biorxiv.org/content/10.1101/357368v1

# LICENSE
# The MIT License (MIT)
# Copyright (c) 2018 Samuel Wolock
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


########## PREPROCESSING PIPELINE


def pipeline_normalize(self, postnorm_total=None):
	''' Total counts normalization '''
	if postnorm_total is None:
		postnorm_total = self._total_counts_obs.mean()

	self._E_obs_norm = tot_counts_norm(self._E_obs, target_total=postnorm_total, total_counts=self._total_counts_obs)

	if self._E_sim is not None:
		self._E_sim_norm = tot_counts_norm(self._E_sim, target_total=postnorm_total, total_counts=self._total_counts_sim)
	return

def pipeline_get_gene_filter(self, min_counts=3, min_cells=3, min_gene_variability_pctl=85):
	''' Identify highly variable genes expressed above a minimum level '''
	self._gene_filter = filter_genes(self._E_obs_norm,
										min_counts=min_counts,
										min_cells=min_cells,
										min_vscore_pctl=min_gene_variability_pctl)
	return

def pipeline_apply_gene_filter(self):
	if self._E_obs is not None:
		self._E_obs = self._E_obs[:,self._gene_filter]
	if self._E_obs_norm is not None:
		self._E_obs_norm = self._E_obs_norm[:,self._gene_filter]
	if self._E_sim is not None:
		self._E_sim = self._E_sim[:,self._gene_filter]
	if self._E_sim_norm is not None:
		self._E_sim_norm = self._E_sim_norm[:,self._gene_filter]
	return

def pipeline_mean_center(self):
	gene_means = self._E_obs_norm.mean(0)
	self._E_obs_norm = self._E_obs_norm - gene_means
	if self._E_sim_norm is not None:
		self._E_sim_norm = self._E_sim_norm - gene_means
	return 

def pipeline_normalize_variance(self):
	gene_stdevs = np.sqrt(sparse_var(self._E_obs_norm))
	self._E_obs_norm = sparse_multiply(self._E_obs_norm.T, 1/gene_stdevs).T
	if self._E_sim_norm is not None:
		self._E_sim_norm = sparse_multiply(self._E_sim_norm.T, 1/gene_stdevs).T
	return 

def pipeline_zscore(self):
	gene_means = self._E_obs_norm.mean(0)
	gene_stdevs = np.sqrt(sparse_var(self._E_obs_norm))
	self._E_obs_norm = np.array(sparse_zscore(self._E_obs_norm, gene_means, gene_stdevs))
	if self._E_sim_norm is not None:
		self._E_sim_norm = np.array(sparse_zscore(self._E_sim_norm, gene_means, gene_stdevs))
	return

def pipeline_log_transform(self, pseudocount=1):
	self._E_obs_norm = log_normalize(self._E_obs_norm, pseudocount)
	if self._E_sim_norm is not None:
		self._E_sim_norm = log_normalize(self._E_sim_norm, pseudocount)
	return

def pipeline_truncated_svd(self, n_prin_comps=30):
	svd = TruncatedSVD(n_components=n_prin_comps).fit(self._E_obs_norm)
	self.set_manifold(svd.transform(self._E_obs_norm), svd.transform(self._E_sim_norm)) 
	return
	
def pipeline_pca(self, n_prin_comps=50):
	if scipy.sparse.issparse(self._E_obs_norm):
		X_obs = self._E_obs_norm.toarray()
	else:
		X_obs = self._E_obs_norm
	if scipy.sparse.issparse(self._E_sim_norm):
		X_sim = self._E_sim_norm.toarray()
	else:
		X_sim = self._E_sim_norm

	pca = PCA(n_components=n_prin_comps).fit(X_obs)
	self.set_manifold(pca.transform(X_obs), pca.transform(X_sim)) 
	return

def matrix_multiply(X, Y):
	if not type(X) == np.ndarray:
		if scipy.sparse.issparse(X):
			X = X.toarray()
		else:
			X = np.array(X)
	if not type(Y) == np.ndarray:
		if scipy.sparse.issparse(Y):
			Y = Y.toarray()
		else:
			Y = np.array(Y)
	return np.dot(X,Y)

def log_normalize(X,pseudocount=1):
	X.data = np.log10(X.data + pseudocount)
	return X

########## LOADING DATA
def load_genes(filename, delimiter='\t', column=0, skip_rows=0):
	gene_list = []
	gene_dict = {}

	with open(filename) as f:
		for iL in range(skip_rows):
			f.readline()
		for l in f:
			gene = l.strip('\n').split(delimiter)[column]
			if gene in gene_dict:
				gene_dict[gene] += 1
				gene_list.append(gene + '__' + str(gene_dict[gene]))
				if gene_dict[gene] == 2:
					i = gene_list.index(gene)
					gene_list[i] = gene + '__1'
			else: 
				gene_dict[gene] = 1
				gene_list.append(gene)
	return gene_list


def make_genes_unique(orig_gene_list):
	gene_list = []
	gene_dict = {}

	for gene in orig_gene_list:
		if gene in gene_dict:
			gene_dict[gene] += 1
			gene_list.append(gene + '__' + str(gene_dict[gene]))
			if gene_dict[gene] == 2:
				i = gene_list.index(gene)
				gene_list[i] = gene + '__1'
		else:
		   gene_dict[gene] = 1
		   gene_list.append(gene)
	return gene_list

########## USEFUL SPARSE FUNCTIONS

def sparse_var(E, axis=0):
	''' variance across the specified axis '''

	mean_gene = E.mean(axis=axis).A.squeeze()
	tmp = E.copy()
	tmp.data **= 2
	return tmp.mean(axis=axis).A.squeeze() - mean_gene ** 2

def sparse_multiply(E, a):
	''' multiply each row of E by a scalar '''

	nrow = E.shape[0]
	w = scipy.sparse.lil_matrix((nrow, nrow))
	w.setdiag(a)
	return w * E

def sparse_zscore(E, gene_mean=None, gene_stdev=None):
	''' z-score normalize each column of E '''

	if gene_mean is None:
		gene_mean = E.mean(0)
	if gene_stdev is None:
		gene_stdev = np.sqrt(sparse_var(E))
	return sparse_multiply((E - gene_mean).T, 1/gene_stdev).T

def subsample_counts(E, rate, original_totals):
	if rate < 1:
		E.data = np.random.binomial(np.round(E.data).astype(int), rate)
		current_totals = E.sum(1).A.squeeze()
		unsampled_orig_totals = original_totals - current_totals
		unsampled_downsamp_totals = np.random.binomial(np.round(unsampled_orig_totals).astype(int), rate)
		final_downsamp_totals = current_totals + unsampled_downsamp_totals
	else:
		final_downsamp_totals = original_totals
	return E, final_downsamp_totals


########## GENE FILTERING

def runningquantile(x, y, p, nBins):

	ind = np.argsort(x)
	x = x[ind]
	y = y[ind]

	dx = (x[-1] - x[0]) / nBins
	xOut = np.linspace(x[0]+dx/2, x[-1]-dx/2, nBins)

	yOut = np.zeros(xOut.shape)

	for i in range(len(xOut)):
		ind = np.nonzero((x >= xOut[i]-dx/2) & (x < xOut[i]+dx/2))[0]
		if len(ind) > 0:
			yOut[i] = np.percentile(y[ind], p)
		else:
			if i > 0:
				yOut[i] = yOut[i-1]
			else:
				yOut[i] = np.nan

	return xOut, yOut


def get_vscores(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
	'''
	Calculate v-score (above-Poisson noise statistic) for genes in the input counts matrix
	Return v-scores and other stats
	'''

	ncell = E.shape[0]

	mu_gene = E.mean(axis=0).A.squeeze()
	gene_ix = np.nonzero(mu_gene > min_mean)[0]
	mu_gene = mu_gene[gene_ix]

	tmp = E[:,gene_ix]
	tmp.data **= 2
	var_gene = tmp.mean(axis=0).A.squeeze() - mu_gene ** 2
	del tmp
	FF_gene = var_gene / mu_gene

	data_x = np.log(mu_gene)
	data_y = np.log(FF_gene / mu_gene)

	x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
	x = x[~np.isnan(y)]
	y = y[~np.isnan(y)]

	gLog = lambda input: np.log(input[1] * np.exp(-input[0]) + input[2])
	h,b = np.histogram(np.log(FF_gene[mu_gene>0]), bins=200)
	b = b[:-1] + np.diff(b)/2
	max_ix = np.argmax(h)
	c = np.max((np.exp(b[max_ix]), 1))
	errFun = lambda b2: np.sum(abs(gLog([x,c,b2])-y) ** error_wt)
	b0 = 0.1
	b = scipy.optimize.fmin(func = errFun, x0=[b0], disp=False)
	a = c / (1+b) - 1


	v_scores = FF_gene / ((1+a)*(1+b) + b * mu_gene);
	CV_eff = np.sqrt((1+a)*(1+b) - 1);
	CV_input = np.sqrt(b);

	return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b

def filter_genes(E, base_ix = [], min_vscore_pctl = 85, min_counts = 3, min_cells = 3, show_vscore_plot = False, sample_name = ''):
	''' 
	Filter genes by expression level and variability
	Return list of filtered gene indices
	'''

	if len(base_ix) == 0:
		base_ix = np.arange(E.shape[0])

	Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(E[base_ix, :])
	ix2 = Vscores>0
	Vscores = Vscores[ix2]
	gene_ix = gene_ix[ix2]
	mu_gene = mu_gene[ix2]
	FF_gene = FF_gene[ix2]
	min_vscore = np.percentile(Vscores, min_vscore_pctl)
	ix = (((E[:,gene_ix] >= min_counts).sum(0).A.squeeze() >= min_cells) & (Vscores >= min_vscore))
	
	if show_vscore_plot:
		import matplotlib.pyplot as plt
		x_min = 0.5*np.min(mu_gene)
		x_max = 2*np.max(mu_gene)
		xTh = x_min * np.exp(np.log(x_max/x_min)*np.linspace(0,1,100))
		yTh = (1 + a)*(1+b) + b * xTh
		plt.figure(figsize=(8, 6));
		plt.scatter(np.log10(mu_gene), np.log10(FF_gene), c = [.8,.8,.8], alpha = 0.3, edgecolors='');
		plt.scatter(np.log10(mu_gene)[ix], np.log10(FF_gene)[ix], c = [0,0,0], alpha = 0.3, edgecolors='');
		plt.plot(np.log10(xTh),np.log10(yTh));
		plt.title(sample_name)
		plt.xlabel('log10(mean)');
		plt.ylabel('log10(Fano factor)');
		plt.show()

	return gene_ix[ix]

########## CELL NORMALIZATION

def tot_counts_norm(E, total_counts = None, exclude_dominant_frac = 1, included = [], target_total = None):
	''' 
	Cell-level total counts normalization of input counts matrix, excluding overly abundant genes if desired.
	Return normalized counts, average total counts, and (if exclude_dominant_frac < 1) list of genes used to calculate total counts 
	'''

	E = E.tocsc()
	ncell = E.shape[0]
	if total_counts is None:
		if len(included) == 0:
			if exclude_dominant_frac == 1:
				tots_use = E.sum(axis=1)
			else:
				tots = E.sum(axis=1)
				wtmp = scipy.sparse.lil_matrix((ncell, ncell))
				wtmp.setdiag(1. / tots)
				included = np.asarray(~(((wtmp * E) > exclude_dominant_frac).sum(axis=0) > 0))[0,:]
				tots_use = E[:,included].sum(axis = 1)
				logging.debug('Excluded %i genes from normalization' %(np.sum(~included)))
		else:
			tots_use = E[:,included].sum(axis = 1)
	else:
		tots_use = total_counts.copy()

	if target_total is None:
		target_total = np.mean(tots_use)

	w = scipy.sparse.lil_matrix((ncell, ncell))
	w.setdiag(float(target_total) / tots_use)
	Enorm = w * E

	return Enorm.tocsc()

########## DIMENSIONALITY REDUCTION

def get_pca(E, base_ix=[], numpc=50, keep_sparse=False, normalize=True):
	'''
	Run PCA on the counts matrix E, gene-level normalizing if desired
	Return PCA coordinates
	'''
	# If keep_sparse is True, gene-level normalization maintains sparsity
	#     (no centering) and TruncatedSVD is used instead of normal PCA.

	if len(base_ix) == 0:
		base_ix = np.arange(E.shape[0])

	if keep_sparse:
		if normalize:
			zstd = np.sqrt(sparse_var(E[base_ix,:]))
			Z = sparse_multiply(E.T, 1 / zstd).T
		else:
			Z = E
		pca = TruncatedSVD(n_components=numpc)

	else:
		if normalize:
			zmean = E[base_ix,:].mean(0)
			zstd = np.sqrt(sparse_var(E[base_ix,:]))
			Z = sparse_multiply((E - zmean).T, 1/zstd).T
		else:
			Z = E
		pca = PCA(n_components=numpc)

	pca.fit(Z[base_ix,:])
	return pca.transform(Z)


def preprocess_and_pca(E, total_counts_normalize=True, norm_exclude_abundant_gene_frac=1, min_counts=3, min_cells=5, min_vscore_pctl=85, gene_filter=None, num_pc=50, sparse_pca=False, zscore_normalize=True, show_vscore_plot=False):
	'''
	Total counts normalize, filter genes, run PCA
	Return PCA coordinates and filtered gene indices
	'''

	if total_counts_normalize:
		logging.debug('Total count normalizing')
		E = tot_counts_norm(E, exclude_dominant_frac = norm_exclude_abundant_gene_frac)[0]

	if gene_filter is None:
		logging.debug('Finding highly variable genes')
		gene_filter = filter_genes(E, min_vscore_pctl=min_vscore_pctl, min_counts=min_counts, min_cells=min_cells, show_vscore_plot=show_vscore_plot)

	logging.debug('Using %i genes for PCA' %len(gene_filter))
	PCdat = get_pca(E[:,gene_filter], numpc=num_pc, keep_sparse=sparse_pca, normalize=zscore_normalize)

	return PCdat, gene_filter

########## GRAPH CONSTRUCTION

def get_knn_graph(X, k=5, dist_metric='euclidean', approx=False, return_edges=True):
	'''
	Build k-nearest-neighbor graph
	Return edge list and nearest neighbor matrix
	'''

	t0 = time.time()
	if approx:
		try:
			from annoy import AnnoyIndex
		except:
			approx = False
			logging.debug('Could not find library "annoy" for approx. nearest neighbor search')
	if approx:
		#logging.debug('Using approximate nearest neighbor search')

		if dist_metric == 'cosine':
			dist_metric = 'angular'
		npc = X.shape[1]
		ncell = X.shape[0]
		annoy_index = AnnoyIndex(npc, metric=dist_metric)

		for i in range(ncell):
			annoy_index.add_item(i, list(X[i,:]))
		annoy_index.build(10) # 10 trees

		knn = []
		for iCell in range(ncell):
			knn.append(annoy_index.get_nns_by_item(iCell, k + 1)[1:])
		knn = np.array(knn, dtype=int)

	else:
		#logging.debug('Using sklearn NearestNeighbors')

		if dist_metric == 'cosine':
			nbrs = NearestNeighbors(n_neighbors=k, metric=dist_metric, algorithm='brute').fit(X)
		else:
			nbrs = NearestNeighbors(n_neighbors=k, metric=dist_metric).fit(X)
		knn = nbrs.kneighbors(return_distance=False)

	if return_edges:
		links = set([])
		for i in range(knn.shape[0]):
			for j in knn[i,:]:
				links.add(tuple(sorted((i,j))))

		t_elapse = time.time() - t0
		#logging.debug('kNN graph built in %.3f sec' %(t_elapse))

		return links, knn
	return knn

def build_adj_mat(edges, n_nodes):
	A = scipy.sparse.lil_matrix((n_nodes, n_nodes))
	for e in edges:
		i, j = e
		A[i,j] = 1
		A[j,i] = 1
	return A.tocsc()

########## 2-D EMBEDDINGS

def get_umap(X, n_neighbors=10, min_dist=0.1, metric='euclidean'):
	import umap
	return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='euclidean').fit_transform(X) 

def get_tsne(X, angle=0.5, perplexity=30, verbose=False):
	from sklearn.manifold import TSNE
	return TSNE(angle=angle, perplexity=perplexity, verbose=verbose).fit_transform(X)

def get_force_layout(X, n_neighbors=5, approx_neighbors=False, n_iter=300, verbose=False):
	edges = get_knn_graph(X, k=n_neighbors, approx=approx_neighbors, return_edges=True)[0]
	return run_force_layout(edges, X.shape[0], verbose=verbose)

def run_force_layout(links, n_cells, n_iter=100, edgeWeightInfluence=1, barnesHutTheta=2, scalingRatio=1, gravity=0.05, jitterTolerance=1, verbose=False):
	from fa2 import ForceAtlas2
	import networkx as nx

	G = nx.Graph()
	G.add_nodes_from(range(n_cells))
	G.add_edges_from(list(links))

	forceatlas2 = ForceAtlas2(
				  # Behavior alternatives
				  outboundAttractionDistribution=False,  # Dissuade hubs
				  linLogMode=False,  # NOT IMPLEMENTED
				  adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
				  edgeWeightInfluence=edgeWeightInfluence,

				  # Performance
				  jitterTolerance=jitterTolerance,  # Tolerance
				  barnesHutOptimize=True,
				  barnesHutTheta=barnesHutTheta,
				  multiThreaded=False,  # NOT IMPLEMENTED

				  # Tuning
				  scalingRatio=scalingRatio,
				  strongGravityMode=False,
				  gravity=gravity,
				  # Log
				  verbose=verbose)

	positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=n_iter)
	positions = np.array([positions[i] for i in sorted(positions.keys())])
	return positions

########## CLUSTERING

def get_spectral_clusters(A, k):
	from sklearn.cluster import SpectralClustering
	spec = SpectralClustering(n_clusters=k, random_state = 0, affinity = 'precomputed', assign_labels = 'discretize')
	return spec.fit_predict(A)


def get_louvain_clusters(nodes, edges):
	import networkx as nx
	import community
	
	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	
	return np.array(list(community.best_partition(G).values()))

########## GENE ENRICHMENT

def rank_enriched_genes(E, gene_list, cell_mask, min_counts=3, min_cells=3, verbose=False):
	gix = (E[cell_mask,:]>=min_counts).sum(0).A.squeeze() >= min_cells
	logging.debug('%i cells in group' %(sum(cell_mask)))
	logging.debug('Considering %i genes' %(sum(gix)))
	
	gene_list = gene_list[gix]
	
	z = sparse_zscore(E[:,gix])
	scores = z[cell_mask,:].mean(0).A.squeeze()
	o = np.argsort(-scores)
	
	return gene_list[o], scores[o]
	

########## PLOTTING STUFF

def darken_cmap(cmap, scale_factor):
	cdat = np.zeros((cmap.N, 4))
	for ii in range(cdat.shape[0]):
		curcol = cmap(ii)
		cdat[ii,0] = curcol[0] * scale_factor
		cdat[ii,1] = curcol[1] * scale_factor
		cdat[ii,2] = curcol[2] * scale_factor
		cdat[ii,3] = 1
	cmap = cmap.from_list(cmap.N, cdat)
	return cmap

def custom_cmap(rgb_list):
	import matplotlib.pyplot as plt
	rgb_list = np.array(rgb_list)
	cmap = plt.cm.Reds
	cmap = cmap.from_list(rgb_list.shape[0],rgb_list)
	return cmap

def plot_groups(x, y, groups, lim_buffer = 50, saving = False, fig_dir = './', fig_name = 'fig', res = 300, close_after = False, title_size = 12, point_size = 3, ncol = 5):
	import matplotlib.pyplot as plt

	n_col = int(ncol)
	ngroup = len(np.unique(groups))
	nrow = int(np.ceil(ngroup / float(ncol)))
	fig = plt.figure(figsize = (14, 3 * nrow))
	for ii, c in enumerate(np.unique(groups)):
		ax = plt.subplot(nrow, ncol, ii+1)
		ix = groups == c

		ax.scatter(x[~ix], y[~ix], s = point_size, c = [.8,.8,.8], edgecolors = '')
		ax.scatter(x[ix], y[ix], s = point_size, c = [0,0,0], edgecolors = '')
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_xlim([min(x) - lim_buffer, max(x) + lim_buffer])
		ax.set_ylim([min(y) - lim_buffer, max(y) + lim_buffer])

		ax.set_title(str(c), fontsize = title_size)

	fig.tight_layout()

	if saving:
		if not os.path.exists(fig_dir):
			os.makedirs(fig_dir)
		plt.savefig(fig_dir + '/' + fig_name + '.png', dpi=res)

	if close_after:
		plt.close()
		

class Scrublet():
	def __init__(self, counts_matrix, total_counts=None, sim_doublet_ratio=2.0, n_neighbors=None, expected_doublet_rate=0.1, stdev_doublet_rate=0.02):
		''' Initialize Scrublet object with counts matrix and doublet prediction parameters

		Parameters
		----------
		counts_matrix : scipy sparse matrix or ndarray, shape (n_cells, n_genes)
			Matrix containing raw (unnormalized) UMI-based transcript counts. 
			Converted into a scipy.sparse.csc_matrix.

		total_counts : ndarray, shape (n_cells,), optional (default: None)
			Array of total UMI counts per cell. If `None`, this is calculated
			as the row sums of `counts_matrix`. 

		sim_doublet_ratio : float, optional (default: 2.0)
			Number of doublets to simulate relative to the number of observed 
			transcriptomes.

		n_neighbors : int, optional (default: None)
			Number of neighbors used to construct the KNN graph of observed
			transcriptomes and simulated doublets. If `None`, this is 
			set to round(0.5 * sqrt(n_cells))

		expected_doublet_rate : float, optional (default: 0.1)
			The estimated doublet rate for the experiment.

		stdev_doublet_rate : float, optional (default: 0.02)
			Uncertainty in the expected doublet rate.

		Attributes
		----------
		predicted_doublets_ : ndarray, shape (n_cells,)
			Boolean mask of predicted doublets in the observed
			transcriptomes. 

		doublet_scores_obs_ : ndarray, shape (n_cells,)
			Doublet scores for observed transcriptomes.

		doublet_scores_sim_ : ndarray, shape (n_doublets,)
			Doublet scores for simulated doublets. 

		doublet_errors_obs_ : ndarray, shape (n_cells,)
			Standard error in the doublet scores for observed
			transcriptomes.

		doublet_errors_sim_ : ndarray, shape (n_doublets,)
			Standard error in the doublet scores for simulated
			doublets.

		threshold_: float
			Doublet score threshold for calling a transcriptome
			a doublet.

		z_scores_ : ndarray, shape (n_cells,)
			Z-score conveying confidence in doublet calls. 
			Z = `(doublet_score_obs_ - threhsold_) / doublet_errors_obs_`

		detected_doublet_rate_: float
			Fraction of observed transcriptomes that have been called
			doublets.

		detectable_doublet_fraction_: float
			Estimated fraction of doublets that are detectable, i.e.,
			fraction of simulated doublets with doublet scores above
			`threshold_`

		overall_doublet_rate_: float
			Estimated overall doublet rate,
			`detected_doublet_rate_ / detectable_doublet_fraction_`.
			Should agree (roughly) with `expected_doublet_rate`.

		manifold_obs_: ndarray, shape (n_cells, n_features)
			The single-cell "manifold" coordinates (e.g., PCA coordinates) 
			for observed transcriptomes. Nearest neighbors are found using
			the union of `manifold_obs_` and `manifold_sim_` (see below).

		manifold_sim_: ndarray, shape (n_doublets, n_features)
			The single-cell "manifold" coordinates (e.g., PCA coordinates) 
			for simulated doublets. Nearest neighbors are found using
			the union of `manifold_obs_` (see above) and `manifold_sim_`.
		
		doublet_parents_ : ndarray, shape (n_doublets, 2)
			Indices of the observed transcriptomes used to generate the 
			simulated doublets.

		doublet_neighbor_parents_ : list, length n_cells
			A list of arrays of the indices of the doublet neighbors of 
			each observed transcriptome (the ith entry is an array of 
			the doublet neighbors of transcriptome i).
		'''

		if not scipy.sparse.issparse(counts_matrix):
			counts_matrix = scipy.sparse.csc_matrix(counts_matrix)
		elif not scipy.sparse.isspmatrix_csc(counts_matrix):
			counts_matrix = counts_matrix.tocsc()

		# initialize counts matrices
		self._E_obs = counts_matrix
		self._E_sim = None
		self._E_obs_norm = None
		self._E_sim_norm = None

		if total_counts is None:
			self._total_counts_obs = self._E_obs.sum(1).A.squeeze()
		else:
			self._total_counts_obs = total_counts

		self._gene_filter = np.arange(self._E_obs.shape[1])
		self._embeddings = {}

		self.sim_doublet_ratio = sim_doublet_ratio
		self.n_neighbors = n_neighbors
		self.expected_doublet_rate = expected_doublet_rate
		self.stdev_doublet_rate = stdev_doublet_rate

		if self.n_neighbors is None:
			self.n_neighbors = int(round(0.5*np.sqrt(self._E_obs.shape[0])))

	######## Core Scrublet functions ########

	def scrub_doublets(self, synthetic_doublet_umi_subsampling=1.0, use_approx_neighbors=True, distance_metric='euclidean', get_doublet_neighbor_parents=False, min_counts=3, min_cells=3, min_gene_variability_pctl=85, log_transform=False, mean_center=True, normalize_variance=True, n_prin_comps=30, verbose=True):
		''' Standard pipeline for preprocessing, doublet simulation, and doublet prediction

		Automatically sets a threshold for calling doublets, but it's best to check 
		this by running plot_histogram() afterwards and adjusting threshold 
		with call_doublets(threshold=new_threhold) if necessary.

		Arguments
		---------
		synthetic_doublet_umi_subsampling : float, optional (defuault: 1.0) 
			Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
			each doublet is created by simply adding the UMIs from two randomly 
			sampled observed transcriptomes. For values less than 1, the 
			UMI counts are added and then randomly sampled at the specified
			rate.

		use_approx_neighbors : bool, optional (default: True)
			Use approximate nearest neighbor method (annoy) for the KNN 
			classifier.

		distance_metric : str, optional (default: 'euclidean')
			Distance metric used when finding nearest neighbors. For list of
			valid values, see the documentation for annoy (if `use_approx_neighbors`
			is True) or sklearn.neighbors.NearestNeighbors (if `use_approx_neighbors`
			is False).
			
		get_doublet_neighbor_parents : bool, optional (default: False)
			If True, return the parent transcriptomes that generated the 
			doublet neighbors of each observed transcriptome. This information can 
			be used to infer the cell states that generated a given 
			doublet state.

		min_counts : float, optional (default: 3)
			Used for gene filtering prior to PCA. Genes expressed at fewer than 
			`min_counts` in fewer than `min_cells` (see below) are excluded.

		min_cells : int, optional (default: 3)
			Used for gene filtering prior to PCA. Genes expressed at fewer than 
			`min_counts` (see above) in fewer than `min_cells` are excluded.

		min_gene_variability_pctl : float, optional (default: 85.0)
			Used for gene filtering prior to PCA. Keep the most highly variable genes
			(in the top min_gene_variability_pctl percentile), as measured by 
			the v-statistic [Klein et al., Cell 2015].

		log_transform : bool, optional (default: False)
			If True, log-transform the counts matrix (log10(1+TPM)). 
			`sklearn.decomposition.TruncatedSVD` will be used for dimensionality
			reduction, unless `mean_center` is True.

		mean_center : bool, optional (default: True)
			If True, center the data such that each gene has a mean of 0.
			`sklearn.decomposition.PCA` will be used for dimensionality
			reduction.

		normalize_variance : bool, optional (default: True)
			If True, normalize the data such that each gene has a variance of 1.
			`sklearn.decomposition.TruncatedSVD` will be used for dimensionality
			reduction, unless `mean_center` is True.

		n_prin_comps : int, optional (default: 30)
			Number of principal components used to embed the transcriptomes prior
			to k-nearest-neighbor graph construction.

		verbose : bool, optional (default: True)
			If True, logging.debug progress updates.

		Sets
		----
		doublet_scores_obs_, doublet_errors_obs_,
		doublet_scores_sim_, doublet_errors_sim_,
		predicted_doublets_, z_scores_ 
		threshold_, detected_doublet_rate_,
		detectable_doublet_fraction_, overall_doublet_rate_,
		doublet_parents_, doublet_neighbor_parents_ 
		'''
		t0 = time.time()

		self._E_sim = None
		self._E_obs_norm = None
		self._E_sim_norm = None
		self._gene_filter = np.arange(self._E_obs.shape[1])

		logging.debug('Preprocessing...')
		pipeline_normalize(self)
		pipeline_get_gene_filter(self, min_counts=min_counts, min_cells=min_cells, min_gene_variability_pctl=min_gene_variability_pctl)
		pipeline_apply_gene_filter(self)

		logging.debug('Simulating doublets...')
		self.simulate_doublets(sim_doublet_ratio=self.sim_doublet_ratio, synthetic_doublet_umi_subsampling=synthetic_doublet_umi_subsampling)
		pipeline_normalize(self, postnorm_total=1e6)
		if log_transform:
			pipeline_log_transform(self)
		if mean_center and normalize_variance:
			pipeline_zscore(self)
		elif mean_center:
			pipeline_mean_center(self)
		elif normalize_variance: 
			pipeline_normalize_variance(self)

		if mean_center:
			logging.debug('Embedding transcriptomes using PCA...')
			pipeline_pca(self, n_prin_comps=n_prin_comps)
		else:
			logging.debug('Embedding transcriptomes using Truncated SVD...')
			pipeline_truncated_svd(self, n_prin_comps=n_prin_comps)            

		logging.debug('Calculating doublet scores...')
		self.calculate_doublet_scores(
			use_approx_neighbors=use_approx_neighbors,
			distance_metric=distance_metric,
			get_doublet_neighbor_parents=get_doublet_neighbor_parents
			)
		self.call_doublets(verbose=verbose)

		t1=time.time()
		logging.debug('Elapsed time: {:.1f} seconds'.format(t1 - t0))
		return self.doublet_scores_obs_, self.predicted_doublets_

	def simulate_doublets(self, sim_doublet_ratio=None, synthetic_doublet_umi_subsampling=1.0):
		''' Simulate doublets by adding the counts of random observed transcriptome pairs.

		Arguments
		---------
		sim_doublet_ratio : float, optional (default: None)
			Number of doublets to simulate relative to the number of observed 
			transcriptomes. If `None`, self.sim_doublet_ratio is used.

		synthetic_doublet_umi_subsampling : float, optional (defuault: 1.0) 
			Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
			each doublet is created by simply adding the UMIs from two randomly 
			sampled observed transcriptomes. For values less than 1, the 
			UMI counts are added and then randomly sampled at the specified
			rate.

		Sets
		----
		doublet_parents_
		'''

		if sim_doublet_ratio is None:
			sim_doublet_ratio = self.sim_doublet_ratio
		else:
			self.sim_doublet_ratio = sim_doublet_ratio

		n_obs = self._E_obs.shape[0]
		n_sim = int(n_obs * sim_doublet_ratio)
		pair_ix = np.random.randint(0, n_obs, size=(n_sim, 2))
		
		E1 = self._E_obs[pair_ix[:,0],:]
		E2 = self._E_obs[pair_ix[:,1],:]
		tots1 = self._total_counts_obs[pair_ix[:,0]]
		tots2 = self._total_counts_obs[pair_ix[:,1]]
		if synthetic_doublet_umi_subsampling < 1:
			self._E_sim, self._total_counts_sim = subsample_counts(E1+E2, synthetic_doublet_umi_subsampling, tots1+tots2)
		else:
			self._E_sim = E1+E2
			self._total_counts_sim = tots1+tots2
		self.doublet_parents_ = pair_ix
		return

	def set_manifold(self, manifold_obs, manifold_sim):
		''' Set the manifold coordinates used in k-nearest-neighbor graph construction

		Arguments
		---------
		manifold_obs: ndarray, shape (n_cells, n_features)
			The single-cell "manifold" coordinates (e.g., PCA coordinates) 
			for observed transcriptomes. Nearest neighbors are found using
			the union of `manifold_obs` and `manifold_sim` (see below).

		manifold_sim: ndarray, shape (n_doublets, n_features)
			The single-cell "manifold" coordinates (e.g., PCA coordinates) 
			for simulated doublets. Nearest neighbors are found using
			the union of `manifold_obs` (see above) and `manifold_sim`.

		Sets
		----
		manifold_obs_, manifold_sim_, 
		'''

		self.manifold_obs_ = manifold_obs
		self.manifold_sim_ = manifold_sim
		return
	
	def calculate_doublet_scores(self, use_approx_neighbors=True, distance_metric='euclidean', get_doublet_neighbor_parents=False):
		''' Calculate doublet scores for observed transcriptomes and simulated doublets

		Requires that manifold_obs_ and manifold_sim_ have already been set.

		Arguments
		---------
		use_approx_neighbors : bool, optional (default: True)
			Use approximate nearest neighbor method (annoy) for the KNN 
			classifier.

		distance_metric : str, optional (default: 'euclidean')
			Distance metric used when finding nearest neighbors. For list of
			valid values, see the documentation for annoy (if `use_approx_neighbors`
			is True) or sklearn.neighbors.NearestNeighbors (if `use_approx_neighbors`
			is False).
			
		get_doublet_neighbor_parents : bool, optional (default: False)
			If True, return the parent transcriptomes that generated the 
			doublet neighbors of each observed transcriptome. This information can 
			be used to infer the cell states that generated a given 
			doublet state.

		Sets
		----
		doublet_scores_obs_, doublet_scores_sim_, 
		doublet_errors_obs_, doublet_errors_sim_, 
		doublet_neighbor_parents_

		'''

		self._nearest_neighbor_classifier(
			k=self.n_neighbors,
			exp_doub_rate=self.expected_doublet_rate,
			stdev_doub_rate=self.stdev_doublet_rate,
			use_approx_nn=use_approx_neighbors, 
			distance_metric=distance_metric,
			get_neighbor_parents=get_doublet_neighbor_parents
			)
		return self.doublet_scores_obs_

	def _nearest_neighbor_classifier(self, k=40, use_approx_nn=True, distance_metric='euclidean', exp_doub_rate=0.1, stdev_doub_rate=0.03, get_neighbor_parents=False):
		manifold = np.vstack((self.manifold_obs_, self.manifold_sim_))
		doub_labels = np.concatenate((np.zeros(self.manifold_obs_.shape[0], dtype=int), 
									  np.ones(self.manifold_sim_.shape[0], dtype=int)))

		n_obs = np.sum(doub_labels == 0)
		n_sim = np.sum(doub_labels == 1)
		
		# Adjust k (number of nearest neighbors) based on the ratio of simulated to observed cells
		k_adj = int(round(k * (1+n_sim/float(n_obs))))
		
		# Find k_adj nearest neighbors
		neighbors = get_knn_graph(manifold, k=k_adj, dist_metric=distance_metric, approx=use_approx_nn, return_edges=False)
		
		# Calculate doublet score based on ratio of simulated cell neighbors vs. observed cell neighbors
		doub_neigh_mask = doub_labels[neighbors] == 1
		n_sim_neigh = doub_neigh_mask.sum(1)
		n_obs_neigh = doub_neigh_mask.shape[1] - n_sim_neigh
		
		rho = exp_doub_rate
		r = n_sim / float(n_obs)
		nd = n_sim_neigh.astype(float)
		ns = n_obs_neigh.astype(float)
		N = float(k_adj)
		
		# Bayesian
		q=(nd+1)/(N+2)
		Ld = q*rho/r/(1-rho-q*(1-rho-rho/r))

		se_q = np.sqrt(q*(1-q)/(N+3))
		se_rho = stdev_doub_rate

		se_Ld = q*rho/r / (1-rho-q*(1-rho-rho/r))**2 * np.sqrt((se_q/q*(1-rho))**2 + (se_rho/rho*(1-q))**2)

		self.doublet_scores_obs_ = Ld[doub_labels == 0]
		self.doublet_scores_sim_ = Ld[doub_labels == 1]
		self.doublet_errors_obs_ = se_Ld[doub_labels==0]
		self.doublet_errors_sim_ = se_Ld[doub_labels==1]

		# get parents of doublet neighbors, if requested
		neighbor_parents = None
		if get_neighbor_parents:
			parent_cells = self.doublet_parents_
			neighbors = neighbors - n_obs
			neighbor_parents = []
			for iCell in range(n_obs):
				this_doub_neigh = neighbors[iCell,:][neighbors[iCell,:] > -1]
				if len(this_doub_neigh) > 0:
					this_doub_neigh_parents = np.unique(parent_cells[this_doub_neigh,:].flatten())
					neighbor_parents.append(this_doub_neigh_parents)
				else:
					neighbor_parents.append([])
			self.doublet_neighbor_parents_ = np.array(neighbor_parents)
		return
	
	def call_doublets(self, threshold=None, verbose=True):
		''' Call trancriptomes as doublets or singlets

		Arguments
		---------
		threshold : float, optional (default: None) 
			Doublet score threshold for calling a transcriptome
			a doublet. If `None`, this is set automatically by looking
			for the minimum between the two modes of the `doublet_scores_sim_`
			histogram. It is best practice to check the threshold visually
			using the `doublet_scores_sim_` histogram and/or based on 
			co-localization of predicted doublets in a 2-D embedding.

		verbose : bool, optional (default: True)
			If True, logging.debug summary statistics.

		Sets
		----
		predicted_doublets_, z_scores_, threshold_,
		detected_doublet_rate_, detectable_doublet_fraction, 
		overall_doublet_rate_
		'''

		if threshold is None:
			# automatic threshold detection
			# http://scikit-image.org/docs/dev/api/skimage.filters.html
			from skimage.filters import threshold_minimum
			try:
				threshold = threshold_minimum(self.doublet_scores_sim_)
				if verbose:
					logging.debug("Automatically set threshold at doublet score = {:.2f}".format(threshold))
			except:
				self.predicted_doublets_ = None
				if verbose:
					logging.debug("Warning: failed to automatically identify doublet score threshold. Run `call_doublets` with user-specified threshold.")
				return self.predicted_doublets_

		Ld_obs = self.doublet_scores_obs_
		Ld_sim = self.doublet_scores_sim_
		se_obs = self.doublet_errors_obs_
		Z = (Ld_obs - threshold) / se_obs
		self.predicted_doublets_ = Ld_obs > threshold
		self.z_scores_ = Z
		self.threshold_ = threshold
		self.detected_doublet_rate_ = (Ld_obs>threshold).sum() / float(len(Ld_obs))
		self.detectable_doublet_fraction_ = (Ld_sim>threshold).sum() / float(len(Ld_sim))
		self.overall_doublet_rate_ = self.detected_doublet_rate_ / self.detectable_doublet_fraction_

		logging.info(f"Estimated doublet fraction {100*self.overall_doublet_rate_:.1f}% (of which {100*self.detectable_doublet_fraction_:.1f}% detectable)")
		return self.predicted_doublets_

	######## Viz functions ########

	def plot_histogram(self, scale_hist_obs='log', scale_hist_sim='linear', fig_size = (8,3)):
		''' Plot histogram of doublet scores for observed transcriptomes and simulated doublets 

		The histogram for simulated doublets is useful for determining the correct doublet 
		score threshold. To set threshold to a new value, T, run call_doublets(threshold=T).

		'''

		fig, axs = plt.subplots(1, 2, figsize = fig_size)

		ax = axs[0]
		ax.hist(self.doublet_scores_obs_, np.linspace(0, 1, 50), color='gray', linewidth=0, density=True)
		ax.set_yscale(scale_hist_obs)
		yl = ax.get_ylim()
		ax.set_ylim(yl)
		ax.plot(self.threshold_ * np.ones(2), yl, c='black', linewidth=1)
		ax.set_title('Observed transcriptomes')
		ax.set_xlabel('Doublet score')
		ax.set_ylabel('Prob. density')

		ax = axs[1]
		ax.hist(self.doublet_scores_sim_, np.linspace(0, 1, 50), color='gray', linewidth=0, density=True)
		ax.set_yscale(scale_hist_sim)
		yl = ax.get_ylim()
		ax.set_ylim(yl)
		ax.plot(self.threshold_ * np.ones(2), yl, c = 'black', linewidth = 1)
		ax.set_title('Simulated doublets')
		ax.set_xlabel('Doublet score')
		ax.set_ylabel('Prob. density')

		fig.tight_layout()

		return fig, axs

	def set_embedding(self, embedding_name, coordinates):
		''' Add a 2-D embedding for the observed transcriptomes '''
		self._embeddings[embedding_name] = coordinates
		return

	def plot_embedding(self, embedding_name, score='raw', marker_size=5, order_points=False, fig_size=(8,4), color_map=None):
		''' Plot doublet predictions on 2-D embedding of observed transcriptomes '''

		#from matplotlib.lines import Line2D
		if embedding_name not in self._embeddings:
			logging.debug('Cannot find "{}" in embeddings. First add the embedding using `set_embedding`.'.format(embedding_name))
			return

		# TO DO: check if self.predicted_doublets exists; plot raw scores only if it doesn't

		fig, axs = plt.subplots(1, 2, figsize = fig_size)

		x = self._embeddings[embedding_name][:,0]
		y = self._embeddings[embedding_name][:,1]
		xl = (x.min() - x.ptp() * .05, x.max() + x.ptp() * 0.05)
		yl = (y.min() - y.ptp() * .05, y.max() + y.ptp() * 0.05)

		ax = axs[1]
		if score == 'raw':
			color_dat = self.doublet_scores_obs_
			vmin = color_dat.min()
			vmax = color_dat.max()
			if color_map is None:
				cmap_use = darken_cmap(plt.cm.Reds, 0.9)
			else:
				cmap_use = color_map
		elif score == 'zscore':
			color_dat = self.z_scores_
			vmin = -color_dat.max()
			vmax = color_dat.max()
			if color_map is None:
				cmap_use = darken_cmap(plt.cm.RdBu_r, 0.9)
			else:
				cmap_use = color_map
		if order_points:
			o = np.argsort(color_dat)
		else:
			o = np.arange(len(color_dat)) 
		pp = ax.scatter(x[o], y[o], s=marker_size, edgecolors='', c = color_dat[o], 
			cmap=cmap_use, vmin=vmin, vmax=vmax)
		ax.set_xlim(xl)
		ax.set_ylim(yl)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title('Doublet score')
		ax.set_xlabel(embedding_name + ' 1')
		ax.set_ylabel(embedding_name + ' 2')
		fig.colorbar(pp, ax=ax)

		ax = axs[0]
		called_doubs = self.predicted_doublets_
		ax.scatter(x[o], y[o], s=marker_size, edgecolors='', c=called_doubs[o], cmap=custom_cmap([[.7,.7,.7], [0,0,0]]))
		ax.set_xlim(xl)
		ax.set_ylim(yl)
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title('Predicted doublets')
		#singlet_marker = Line2D([], [], color=[.7,.7,.7], marker='o', markersize=5, label='Singlet', linewidth=0)
		#doublet_marker = Line2D([], [], color=[.0,.0,.0], marker='o', markersize=5, label='Doublet', linewidth=0)
		#ax.legend(handles = [singlet_marker, doublet_marker])
		ax.set_xlabel(embedding_name + ' 1')
		ax.set_ylabel(embedding_name + ' 2')

		fig.tight_layout()

		return fig, axs



	

