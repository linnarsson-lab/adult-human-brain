import matplotlib.pyplot as plt
import numpy as np
import loompy
import os

def mito_genes_ratio(ds: loompy.LoomConnection) -> None:
    mito_genes = np.where(ds.ra.Chromosome=="chrM")[0]
    exp_mito = ds[mito_genes,:]
    sum_mito = exp_mito.sum(axis=0)
    sum_all = ds.ca.TotalUMI
    mito_ratio = np.divide(sum_mito,sum_all)
    ds.ca["MT_ratio"] = mito_ratio
    
        
def unspliced_ratio(ds: loompy.LoomConnection, graphs: bool = True, sample_name : object = "tmp") -> None:
    u = ds.layers["unspliced"][:]
    sum_all = ds.ca.TotalUMI
    sum_u = u.sum(axis=0)
    unspliced_ratio = np.divide(sum_u,sum_all)
    ds.ca["unspliced_ratio"] = unspliced_ratio


