from sklearn.cluster import KMeans as sk_KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.special import polygamma
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import pearsonr, ks_2samp, binom
from scipy.spatial.distance import cdist
from pylab import * # CRAAAAAAZY CRAZY CRAZY SPEEDUP
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
        Input data loaded in memory
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
        X = X / np.sqrt( np.sum(X**2, 0) ) # divide by unnormalized standard deviation over axis=0
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
                    updated_centroids[i,:] = X[np.argmax( np.min(D,1) ), :]

                if metric == "correlation":
                    # This bit is taken from MATLAB source code.
                    # The rationale is that the centroids should be recentered 
                    updated_centroids = updated_centroids - updated_centroids.mean(1)[:,None]

            # If all the centroids are not uppdated (within a max tolerance) Stop updating
            if np.all(paired_distances(centroids, updated_centroids, metric=metric_f) < tol, 0):
                break
        # Calculate inertia and keep track of the iteration with smallest inertia
        inertia = np.sum( D[np.arange(X.shape[0]), label] )
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
			pca = PCA(n_components=n_components)

			if current_cells_ixs.shape[0] > cell_limit:
				selection = np.random.choice(np.arange(current_cells_ixs.shape[0]), cell_limit, replace=False)
				pca.fit( data_tmp[:,selection].T )
			else:
				pca.fit( data_tmp.T )
			
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
				labels = sk_KMeans(n_clusters=2, n_init=3, n_jobs=1).fit_predict(data_tmp.T)
				
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


def amit_biPCA(data, n_splits=10, n_components=100, cell_limit=10000, smallest_allowed = 5, verbose=2):
	'''biPCA algorythm for clustering using PCA splits

	Args
	----
		data : np.array (genes, cells)
			Data is assumed to be in row molecule counts format
			with genes already selected. 
		n_splits : int
			number of splits to be attempted
		n_components: int
			max number of principal components
		cell_limit : int
			the max number of cells that are used to calculate PCA, if #cells>cell_limit, 
			a random sample of cell_limit cells will be drawn
		smallest_allowed: int
			the size of the smallest clsuter allowed, split that generates a cluster of this size is rejected
		level_of_verbosity: int
			0 for only ERRORS
			1 for WARNINGS and ERRORS
			2 for INFOS, WARNINGS and ERRORS
			3 for DEBUGGING, NFOS, WARNINGS and ERRORS

	Returns
	-------
		labels_matrix : np.array (split_depth, labels)
			The assigned cluster at every iteration of the algorythm

	'''

	logger = logging.getLogger('biPCA_logger')
	logger.setLevel([logging.ERROR,logging.WARNING,logging.INFO,logging.DEBUG][verbose])
	ch = logging.StreamHandler()
	formatter = logging.Formatter('%(message)s')
	ch.setFormatter( formatter )
	logger.handlers = []
	logger.addHandler(ch)

	n_genes, n_cells = data.shape
	cell_labels_by_depth = np.zeros((n_splits+1, n_cells))
	
	# Run an iteration per level of depth
	for i_split in range(n_splits):
		logger.info( "Depth: %i" % i_split )
		running_label_id = 0
		parent_clusters = np.unique(cell_labels_by_depth[i_split, :])
		# Consider every parent cluster and split them one by one
		for parent in parent_clusters:
			# Select the current cell cluster
			logger.debug( "Log-normalize the data" )
			current_cells_ixs = np.where( cell_labels_by_depth[i_split, :] == parent )[0]
			data_tmp = np.log2( data[:,current_cells_ixs] + 1)
			data_tmp -= data_tmp.mean(1)[:,None]

			# Perform PCA
			logger.debug( "Performing PCA" )
			pca = PCA(n_components=n_components)
			if current_cells_ixs.shape[0] > cell_limit:
				selection = np.random.choice(np.arange(current_cells_ixs.shape[0]), cell_limit, replace=False)
				pca.fit( data_tmp[:,selection].T )
			else:
				pca.fit( data_tmp.T )
			data_tmp = pca.transform( data_tmp.T ).T # This is the pca projection from now on

			# Select significant principal components by KS test, this is more conservative than broken stick
			pvalue_KS = zeros(data_tmp.shape[0]) # pvalue of each component
			for i in range(1,data_tmp.shape[0]):
				[_, pvalue_KS[i]] = ks_2samp(data_tmp[i-1,:],data_tmp[i,:])
			
			# The following lines have been fixed and differ from the original implementation Amit
			# Amit does: sig = pvalue_KS < 0.1
			# This is wrong becouse one should stop after you find the first nonsignificant component
			first_not_sign = where(pvalue_KS>0.1)[0][0]
			sig = zeros_like(pvalue_KS,dtype=bool) 
			sig[:first_not_sign] = True
			
			# If the two first pcs are not significant: don't split
			if sum(sig)<2:
				logger.debug( "Two first pcs are not significant: don't split." )
				cell_labels_by_depth[i_split+1, current_cells_ixs] = running_label_id
				running_label_id += 1
				continue
			logger.debug('%i principal components are significant' % sum(sig))

			# Drop the not significant PCs
			data_tmp = data_tmp[sig, : ]

			# Perform KMEANS clustering
			best_labels = kmeans(data_tmp.T, k=2, metric="correlation", n_iter=10) # TODO This could be parallelized 
			logger.debug("Proposed split (%i,%i)" % (sum(best_labels==0),sum(best_labels==1)) )
			ids, cluster_counts = np.unique(best_labels, return_counts=True)			
			
			# Check that no small cluster gets generated
			if np.min(cluster_counts) < smallest_allowed:
				# Reject split immediatelly and continue
				logger.debug( "A small cluster get generated, don't split'" )
				cell_labels_by_depth[i_split+1, current_cells_ixs] = running_label_id
				running_label_id += 1
				continue
			
			# Consider only the 500 genes with top loadings
			sum_loadings_per_gene =  np.abs(  pca.components_.T[:,:first_not_sign].sum(1) )
			topload_gene_ixs = np.argsort(sum_loadings_per_gene)[::-1][:500]
			
			# Test their significance using a Binomial test
			# I changed the Amit implementation noting that some of the line are true 
			# only true when sum(best_labels==0) == sum(best_labels==1)
			# For exmaple in the case of split of 75 cells in (18 and 57)
			# binom.cdf(18,50,0.5) ~= 0.03 and binom.cdf( 57, 100, 0.500 ) ~= 0.9
			# For more strictness consider hypergeometric
			cells_pos_group1 = np.sum( data[ topload_gene_ixs[:,None], current_cells_ixs[ best_labels == 0 ]] > 0, 1 )
			cells_pos_group2 = np.sum( data[ topload_gene_ixs[:,None], current_cells_ixs[ best_labels == 1 ]] > 0, 1 )
			prob = (cells_pos_group1+cells_pos_group2) / len(current_cells_ixs)
			pvalue_binomial1 = binom.cdf(cells_pos_group1, np.sum(best_labels == 0), prob )
			pvalue_binomial1 = np.minimum(pvalue_binomial1, 1-pvalue_binomial1)
			pvalue_binomial2 = binom.cdf(cells_pos_group2, np.sum(best_labels == 1), prob )
			pvalue_binomial2 = np.minimum(pvalue_binomial2, 1-pvalue_binomial2)
			pvalue_binomial = np.minimum(pvalue_binomial1, pvalue_binomial2)
			rejected_null, q_values = multipletests(pvalue_binomial, 0.01, 'fdr_bh')[:2]

			# Decide if we should accept the split
			# Here instead of arbitrary threshold on shilouette I could boostrap
			scores_percell = silhouette_samples(data_tmp.T, best_labels, "correlation")
			s_score = np.minimum( np.mean(scores_percell[best_labels==0]), np.mean(scores_percell[best_labels==1]) )
			if (np.sum(rejected_null) > 5) and (s_score>0.1):
				# Accept
				logger.debug("Spltting with significant genes: %s; silhouette-score: %s" % (np.sum(rejected_null), s_score))
				cell_labels_by_depth[i_split+1, current_cells_ixs[best_labels == 0]] = running_label_id 
				cell_labels_by_depth[i_split+1, current_cells_ixs[best_labels == 1]] = running_label_id + 1
				running_label_id += 2
			else:
				# Reject
				logger.debug( "Don't split. Significant genes: %s (min=5); silhouette-score: %s(min=0.01)" % (np.sum(rejected_null), s_score) )
				cell_labels_by_depth[i_split+1, current_cells_ixs] = running_label_id
				running_label_id += 1

	return cell_labels_by_depth




def test_gene(ds, cells, gene_ix, label, group):
    b = np.log2(ds[gene_ix,:][cells]+1)[label != group]
    a = np.log2(ds[gene_ix,:][cells]+1)[label == group]
    (_,pval) = mannwhitneyu(a,b,alternative="greater")
    return pval


__original_MATLAB_code__ = """
function [dataout_sorted,cells_groups,cellsorder,cells_bor,gr_cells_vec] = biPCAsplit_v4(data, n_splits, n_pca, min_gr);

[N,M] = size(data);
gr_cells_vec = ones(M,n_splits+1);

for i=1:n_splits
    i
    k = 0;
    gr_uni = unique(gr_cells_vec(:,i));
    for j=1:length(gr_uni);
        cells_curr = find(gr_cells_vec(:,i)==j);
        if length(cells_curr)>min_gr
            datalog_tmp = cent_norm([(log2(data(:,cells_curr)+1))]);
            if length(cells_curr)<1e4
                [U]=pcasecon(datalog_tmp,n_pca);
            else
                [U]=pcasecon(datalog_tmp(:,[1:round(length(cells_curr)/1e4):length(cells_curr)]),n_pca);
            end
            prj = [U'*datalog_tmp]';
            if prj~=-1
                pks = zeros(n_pca,1);
                for jj=2:n_pca
                    [~,pks(jj)] = kstest2(prj(:,jj-1),prj(:,jj));
                end
                prj = prj(:,pks<0.1);
                length(prj(1,:))
                if length(prj(1,:))>2
                    idx = zeros(length(prj(:,1)),5);
                    for jjj=1:5;
                        idx(:,jjj) = kmeans(prj,2,'distance','correlation');
                        s = silhouette(prj,idx(:,jjj),'correlation');
                        %             figure(111);
                        %             kk=1; plot3(prj(idx==kk,1),prj(idx==kk,2),prj(idx==kk,3),'.r');hold on
                        %             kk=2; plot3(prj(idx==kk,1),prj(idx==kk,2),prj(idx==kk,3),'.c');
                        %             close(111)
                        min_s(jjj) = min([mean(s(idx(:,jjj)==1)),mean(s(idx(:,jjj)==2))]);
                        sum(s<0.1)/length(s)
                    end
                    [~,imin] = max(min_s);
                    idx = idx(:,imin);
                    [~,top_load_genes] = sort(sum(abs(U),2),'descend');
                    top_load_genes = top_load_genes(1:500);
                    n_gr1 = sum(data(top_load_genes,cells_curr(idx==1))>0,2);
                    n_gr2 = sum(data(top_load_genes,cells_curr(idx==2))>0,2);
                    %                     fold_pos = n_gr1./(n_gr1+n_gr2);
                    %                     fold_pos(fold_pos<0.5) = 1 - fold_pos(fold_pos<0.5);
                    %                     [fold_possort,fold_xi] = sort(fold_pos,'descend');
                    pbin = binocdf(n_gr1,length(cells_curr),(n_gr1+n_gr2)/length(cells_curr));
                    pbin = 2*min([pbin,1-pbin],[],2);
                    length(fdr_proc(pbin,0.05))
                    if min_s(imin)>0.1 & length(fdr_proc(pbin,0.01))>10
                        gr_cells_vec(cells_curr(idx==1),i+1) = k+1;
                        gr_cells_vec(cells_curr(idx==2),i+1) = k+2;
                        k = k+2;
                    else
                        gr_cells_vec(cells_curr,i+1) = k+1;
                        k = k+1;
                    end
                else
                    gr_cells_vec(cells_curr,i+1) = k+1;
                    k = k+1;
                end
            else
                gr_cells_vec(cells_curr,i+1) = k+1;
                k = k+1;
            end
        else
            gr_cells_vec(cells_curr,i+1) = k+1;
            k = k+1;
        end
    end
end

[cells_groups,cellsorder] = sort(gr_cells_vec(:,end)');
cells_bor = find(diff(cells_groups)>0)+1;
gr_cells_vec = gr_cells_vec(cellsorder,:);
dataout_sorted = data(:,cellsorder);

T_cells_tmp = gr_cells_vec(:,end);
T_cells_tmp_uni = unique(T_cells_tmp);
meanpergene = mean(data,2);%
molenrich_mat = zeros(length(data(:,1)),length(T_cells_tmp_uni));
meangr_mat = zeros(length(data(:,1)),length(T_cells_tmp_uni));
meangrpos_mat = zeros(length(data(:,1)),length(T_cells_tmp_uni));
for jjj=1:length(T_cells_tmp_uni)
    jjj
    %         gr_center(jjj) = mean(find(T_cells_tmp==T_cells_tmp_uni(jjj)));
    molenrich_mat(:,jjj) = mean(dataout_sorted(:,T_cells_tmp==T_cells_tmp_uni(jjj)),2)./meanpergene;
    meangrpos_mat(:,jjj) = mean(dataout_sorted(:,T_cells_tmp==T_cells_tmp_uni(jjj))>0,2);
end
molenrich_mat(meanpergene==0,:) = 0;
meangrpos_mat(meangrpos_mat<0.1) = 0;
molenrich_mat(meanpergene==0 | isnan(meanpergene),:) = 0;
[sc_0p5,xi0p5] = sort(molenrich_mat.*meangrpos_mat.^0.5,'descend');
[sc_1,xi1] = sort(molenrich_mat.*meangrpos_mat.^1,'descend');

num_per_power = 10;
ind_gr_tmp_mark = [xi0p5(1:num_per_power,:);xi1(1:num_per_power,:)];%xi0(1:100,:);
sc_gr_tmp_mark = [sc_0p5(1:num_per_power,:);sc_1(1:num_per_power,:)];
ind_gr_tmp_mark = flipud(ind_gr_tmp_mark(:));
sc_gr_tmp_mark = flipud(sc_gr_tmp_mark(:));

[~,xi] = sortrows([ind_gr_tmp_mark,sc_gr_tmp_mark],[1,2]);
ind_gr_tmp_mark = ind_gr_tmp_mark(xi);
sc_gr_tmp_mark = sc_gr_tmp_mark(xi);
[~,ia] = unique(ind_gr_tmp_mark,'last');
ind_gr_tmp_mark = ind_gr_tmp_mark(ia);
sc_gr_tmp_mark = sc_gr_tmp_mark(ia);
ind_gr_tmp_mark(sc_gr_tmp_mark==0) = [];

dataout_sorted = data(ind_gr_tmp_mark,cellsorder);


% % % % % % % % % % % % % % % % % % % % % % % % % % order the blocks such
% that similar are closer
gr_uni = T_cells_tmp_uni;
num_gr = length(gr_uni);
med_exp = zeros(length(ind_gr_tmp_mark),num_gr);
for i=1:num_gr % calculate the median expression for each group
    med_exp(:,i) = median(dataout_sorted(:,T_cells_tmp==gr_uni(i)),2);
end
empty_cluster = find(sum(med_exp)==0);
gr_cellorder = find(sum(med_exp)>0);
med_exp = med_exp(:,gr_cellorder);
Z = linkage(log2(med_exp+1)','average','correlation');
D = pdist(log2(med_exp+1)','correlation');
% D = squareform(1-corr_mat(log2(med_exp+1)));
% Z = linkage(D,'average');
leaforder = optimalleaforder(Z,D);
gr_cellorder = [gr_cellorder(leaforder),empty_cluster];%1:num_gr;
T_cells_new = zeros(size(T_cells_tmp));
for i=1:length(gr_cellorder)
    T_cells_new(T_cells_tmp==gr_cellorder(i)) = i;
end

[T_cells_tmp,XIX] = sort(T_cells_new);
cellsorder = cellsorder(XIX);
cells_groups = T_cells_tmp;
cells_bor = find(diff(cells_groups)>0)+1;
gr_cells_vec = gr_cells_vec(XIX,:);
dataout_sorted = dataout_sorted(:,XIX);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

T_cells_tmp_uni = unique(T_cells_tmp);
meanpergene = mean(dataout_sorted,2);%
molenrich_mat = zeros(length(dataout_sorted(:,1)),length(T_cells_tmp_uni));
meangr_mat = zeros(length(dataout_sorted(:,1)),length(T_cells_tmp_uni));
meangrpos_mat = zeros(length(dataout_sorted(:,1)),length(T_cells_tmp_uni));
for jjj=1:length(T_cells_tmp_uni)
    jjj
    %         gr_center(jjj) = mean(find(T_cells_tmp==T_cells_tmp_uni(jjj)));
    molenrich_mat(:,jjj) = mean(dataout_sorted(:,T_cells_tmp==T_cells_tmp_uni(jjj)),2)./meanpergene;
    meangrpos_mat(:,jjj) = mean(dataout_sorted(:,T_cells_tmp==T_cells_tmp_uni(jjj))>0,2);
end
molenrich_mat(meanpergene==0,:) = 0;
meangrpos_mat(meangrpos_mat<0.1) = 0;
molenrich_mat(meanpergene==0 | isnan(meanpergene),:) = 0;
[sc_0p5,xi0p5] = sort(molenrich_mat.*meangrpos_mat.^0.5,'descend');
[sc_1,xi1] = sort(molenrich_mat.*meangrpos_mat.^1,'descend');

num_per_power = 10;
ind_gr_tmp_mark = [xi0p5(1:num_per_power,:);xi1(1:num_per_power,:)];%xi0(1:100,:);
sc_gr_tmp_mark = [sc_0p5(1:num_per_power,:);sc_1(1:num_per_power,:)];
ind_gr_tmp_mark = flipud(ind_gr_tmp_mark(:));
sc_gr_tmp_mark = flipud(sc_gr_tmp_mark(:));

[~,xi] = sortrows([ind_gr_tmp_mark,sc_gr_tmp_mark],[1,2]);
ind_gr_tmp_mark = ind_gr_tmp_mark(xi);
sc_gr_tmp_mark = sc_gr_tmp_mark(xi);
[~,ia] = unique(ind_gr_tmp_mark,'last');
ind_gr_tmp_mark = ind_gr_tmp_mark(ia);
sc_gr_tmp_mark = sc_gr_tmp_mark(ia);
ind_gr_tmp_mark(sc_gr_tmp_mark==0) = [];

dataout_sorted = dataout_sorted(ind_gr_tmp_mark,cellsorder);
"""