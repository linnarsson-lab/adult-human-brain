import loompy
from sklearn.decomposition import PCA
import numpy as np
import pickle

def permute_pca(ds, n_comps, max_cells=None, donor=None):

    """
    Calculate PCA on a dataset before and after permuting gene counts across cells
    ds: a loom connection
    n_comps: number of components to calculate
    max_cells: subsample the dataset to less than max_cells before analysis
    donor: calculate PCA on cells from a specific donor
    """

    print('Loading data')
    # use genes that have been previously selected
    # to analyze the dataset: in Siletti et. al,
    # highly variable genes
    selected_genes = ds.ra.Selected == 1
    # default is to select all cells
    selected_cells = np.arange(ds.shape[1])
    # if donor is specified, select donor's cells
    if donor is not None:
        print(f'Analyzing donor {donor}')
        selected_cells = np.where(ds.ca.Donor == donor)[0]
    # if max_cells is specified      
    if max_cells is not None:
        # and more than max_cells were selected
        if len(selected_cells) > max_cells:
            # downsample
            selected_cells = np.random.choice(selected_cells, size=max_cells, replace=False)
    print(f'Selecting {len(selected_cells)} cells')
    selected_cells = np.sort(selected_cells)
    # load data
    data = ds.sparse(selected_genes, selected_cells).A
    # normalize to median total UMI count
    totals = ds.ca.TotalUMI[selected_cells]
    data_norm = np.log2(data / totals * np.median(totals) + 1)
    print('Fitting PCA')
    pca = PCA(n_components=n_comps)
    comps = pca.fit_transform(data_norm.T)
    exp_var = pca.explained_variance_ratio_

    print('Shuffling gene counts across cells')
    data_norm_random = np.zeros(data_norm.shape)
    n_genes = data_norm.shape[0]
    n_cells = data_norm.shape[1]
    for i in range(n_genes):
        shuffled = np.random.permutation(n_cells)
        data_norm_random[i, :] = data_norm[i, shuffled]

    print('Fitting PCA to shuffled data')
    pca = PCA(n_components=n_comps)
    comps_permuted = pca.fit_transform(data_norm_random.T)
    exp_var_permuted = pca.explained_variance_ratio_
    
    return comps, comps_permuted, exp_var, exp_var_permuted, selected_cells


with loompy.connect('/proj/human_adult/20220222/harmony/paris_top_bug/data/Pool.loom', 'r') as ds:
    punchcards = np.unique(ds.ca.Punchcard)

# permute each donor separately
for donor in ['H18.30.002', 'H19.30.001', 'H19.30.002']:

    print(f'Initializing')
    
    # inititalize dictionaries
    comps = {}
    comps_permuted = {}
    exp_var = {}
    exp_var_permuted = {}
    selected = {}

    # loop through punchcards
    for subset in punchcards:
        print(f'\n{subset}')
        with loompy.connect(f'/proj/human_adult/20220222/harmony/paris_top_bug/data/{subset}.loom', 'r') as ds:
                        
            # calculate unpermuted and permuted PCAs
            comps[subset], comps_permuted[subset], exp_var[subset], exp_var_permuted[subset], selected[subset] = permute_pca(
                ds, 
                n_comps=50,
                max_cells=10_000,
                donor=donor
            )

    # add serotonergic neurons
    subset = 'harmony_A_A_DDDD'
    with loompy.connect(f'/proj/human_adult/20220222/harmony/by_cluster/data/{subset}.loom', 'r') as ds:
        # calculate unpermuted and permuted PCAs
        comps[subset], comps_permuted[subset], exp_var[subset], exp_var_permuted[subset], selected[subset] = permute_pca(
            ds, 
            n_comps=50,
            max_cells=10_000,
            donor=donor
        )

   # add dopaminergic neurons
    subset = 'harmony_A_A_EEEE'
    with loompy.connect(f'/proj/human_adult/20220222/harmony/by_cluster/data/{subset}.loom', 'r') as ds:
        # calculate unpermuted and permuted PCAs
        comps[subset], comps_permuted[subset], exp_var[subset], exp_var_permuted[subset], selected[subset] = permute_pca(
            ds, 
            n_comps=50,
            max_cells=10_000,
            donor=donor
        )

    # dump results for donor
    pickle.dump(comps, open(f'permutation_{donor}_comps.p', 'wb'))
    pickle.dump(comps_permuted, open(f'permutation_{donor}_comps_permuted.p', 'wb'))
    pickle.dump(exp_var, open(f'permutation_{donor}_exp_var.p', 'wb'))
    pickle.dump(exp_var_permuted, open(f'permutation_{donor}_exp_var_permuted.p', 'wb'))
    pickle.dump(selected, open(f'permutation_{donor}_selected.p', 'wb'))
