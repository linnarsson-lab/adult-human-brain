import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import loompy
from .colors import colorize
from cytograph.enrichment import FeatureSelectionByVariance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def dist_attr(ds: loompy.LoomConnection, out_file: str, attr: str, plot_title: str = None ,line: float = None) -> None:

    if attr in ds.ca:
        fig = plt.figure(figsize=(12,8))
        plt.hist(ds.ca[attr],bins = 100)
        plt.xlim(np.amin(ds.ca[attr]),np.amax(ds.ca[attr]))
        if line is not None:
            plt.axvline(x=line, c='r')
        if plot_title is not None:
            plt.title(plot_title)
        plt.savefig(out_file, dpi=144)
        plt.close()


def attrs_on_TSNE(ds: loompy.LoomConnection, out_file: str, attrs: list, plot_title: list = None )-> None:
    n_attr = len(attrs)
    if(n_attr>=2):
        n_cols = 2
    else:
        n_cols = 1
    n_rows = np.ceil(n_attr/n_cols)
    plt.figure(figsize=(12, 12))
    
    if 'TSNE' in ds.ca:
        xy = ds.ca.TSNE
    elif 'HPF' in ds.ca:
        xy = tsne(ds.ca.HPF)
        ds.ca.TSNE = xy
    elif 'PCA' in ds.ca:
        angle=0.5
        perplexity=30
        verbose=False
        xy = TSNE(angle=angle, perplexity=perplexity, verbose=verbose).fit_transform(ds.ca.PCA)
        ds.ca.TSNE = xy
    else: 

        genes = FeatureSelectionByVariance(2000).fit(ds)
        data = ds[:, :]
        f = np.divide(data.sum(axis=0), 10e6)
        norm_data = np.divide(data, f)
        norm_data = np.log(norm_data + 1)
        ds.ca.PCA = PCA(n_components=50).fit_transform(norm_data[genes, :].T)
        angle=0.5
        perplexity=30
        verbose=False
        xy = TSNE(angle=angle, perplexity=perplexity, verbose=verbose).fit_transform(ds.ca.PCA)
        ds.ca.TSNE = xy
    
    for n,attr in enumerate(attrs):
        ax = plt.subplot(n_rows,n_cols,n+1)
        if attr in ds.ca:
            
            ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', lw=0, s=10)
            labels = ds.ca[attr]
            if (len(np.unique(labels))>20):
                if(np.max(ds.ca[attr])>10000):
                    labels = np.log(labels)
                    if plot_title is not None:
                        plt.title(plot_title[n]+" (log transformed)")
                elif plot_title is not None:
                    plt.title(plot_title[n])
                cm = plt.cm.get_cmap('jet')
                cells = ds.ca[attr] > 0
                sc = ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=labels[cells], lw=0, s=10,cmap = cm)
                plt.colorbar(sc,ax=ax)
            else:
                for lbl in np.unique(labels):
                    cells = labels == lbl
                    ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=colorize(labels)[cells], label=lbl, lw=0, s=0)
                    cells = np.random.permutation(labels.shape[0])
                    ax.scatter(xy[:, 0][cells], xy[:, 1][cells], c=colorize(labels)[cells], lw=0, s=10)
                    lgnd = ax.legend()
                    if plot_title is not None:
                    #Add min, max
                        plt.title(plot_title[n])
                    for handle in lgnd.legendHandles:
                        handle.set_sizes([10])
            
            
          
        else:
            ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', lw=0, s=10)
            if plot_title is not None:
                plt.title(plot_title[n]) 
    plt.savefig(out_file, dpi=144)
    plt.close()
    return()       
            
           



    
        