
import loompy
from sklearn.decomposition import PCA
import numpy as np
import pickle

def permuted_pca(ds, n_comps, n_cells=None, cluster=None):

    print('Loading data')
    selected_genes = ds.ra.Selected == 1
    if n_cells is not None:
        selected_cells = np.random.choice(ds.shape[1], size=n_cells, replace=False)
    elif cluster is not None:
        selected_cells = ds.ca.Clusters == cluster
        selected_cells = np.sort(selected_cells)
    else:
        print('Using all cells')
        selected_cells = np.arange(ds.shape[1])
    data = ds.sparse(selected_genes, selected_cells).A
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
    
    return comps, comps_permuted, exp_var, exp_var_permuted, ds.ca.TSNE[selected_cells]


with loompy.connect('/proj/human_adult/20220222/harmony/paris_top_bug/data/Pool.loom', 'r') as ds:
    punchcards = np.unique(ds.ca.Punchcard)
    
comps = {}
comps_permuted = {}
exp_var = {}
exp_var_permuted = {}
xy = {}

max_cells = 50_000
n_cells = []

for subset in punchcards:
    print(f'\n{subset}')
    with loompy.connect(f'/proj/human_adult/20220222/harmony/paris_top_bug/data/{subset}.loom', 'r') as ds:
        if ds.shape[1] < max_cells:
            comps[subset], comps_permuted[subset], exp_var[subset], exp_var_permuted[subset], xy[subset] = permuted_pca(
                ds, 
                n_comps=50, 
            )
            n_cells.append(ds.shape[1])
        else:
            comps[subset], comps_permuted[subset], exp_var[subset], exp_var_permuted[subset], xy[subset] = permuted_pca(
                ds, 
                n_comps=50, 
                n_cells=max_cells
            )
            n_cells.append(50_000)

pickle.dump(comps, open('comps.p', 'wb'))
pickle.dump(comps_permuted, open('comps_permuted.p', 'wb'))
pickle.dump(exp_var, open('exp_var.p', 'wb'))
pickle.dump(exp_var_permuted, open('exp_var_permuted.p', 'wb'))
pickle.dump(xy, open('xy.p', 'wb'))
