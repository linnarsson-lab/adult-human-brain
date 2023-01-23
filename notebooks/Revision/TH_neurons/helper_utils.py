import numpy as np



def ixs_thatsort_a2b(a: np.ndarray, b: np.ndarray, check_content: bool=True) -> np.ndarray:
    "This is super duper magic sauce to make the order of one list to be like another"
    if check_content:
        assert len(np.intersect1d(a, b)) == len(a), f"The two arrays are not matching"
    return np.argsort(a)[np.argsort(np.argsort(b))]


def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c



def inter_gene(a,b,return_one_indices_a=True):
    
    """
    try to match b to a!!
    """
    b_ = np.vstack([b,b]).T
    a_ = np.vstack([a,a]).T
    
    _,iloc_a,iloc_b = np.intersect1d(a_[:,0],b_[:,0],return_indices=True)

    gene_inter = a_[iloc_a,:] # matching genes between a and b

    stack = np.vstack((gene_inter,a_[~np.in1d(a_[:,0],gene_inter[:,0])]*[1,0])) #combine matching genes and missing one
    iloc = ixs_thatsort_a2b(stack[:,0],a_[:,0]) # sort compiled genes to a 
    
    b_inter_gene = stack[iloc,:][:,1]
    match_b = np.where(b_inter_gene!='')[0]
    iloc_b_ = ixs_thatsort_a2b(gene_inter[:,0],b_inter_gene[match_b])
    
    b_inter_iloc = iloc_b[iloc_b_]# (iloc_b,iloc_b_) #one_iloc_b[0][one_iloc_b[1]]

    
    if return_one_indices_a:
        indices = np.where(stack[iloc,:][:,1]!='')[0] # b match for a - a.iloc equivalent
        
        return b_inter_gene,indices,b_inter_iloc
    
    else:
        return b_inter_gene


