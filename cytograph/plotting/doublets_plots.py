import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import loompy
from ..embedding import tsne
from .colors import colorize
from sklearn.manifold import TSNE
import os

def plot_all (ds: loompy.LoomConnection, out_file: str, labels: np.array = None,  doublet_score_A: np.array = None,logprob:np.array  = None, xx: np.array = None, score1: float = 1, score2: float =1 ,score: float = 1)->None:
       
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,  figsize=(12,12))
    doublets_TSNE(ax1, ds,labels)
    fake_doublets_dist(ax2,doublet_score_A,logprob, xx, score1, score2,score)
    doublets_umis(ax3,ds, labels)
    doublets_ngenes(ax4, ds, labels)
    
    f.savefig(out_file, dpi=144)

def doublets_TSNE( ax:plt.axes = None, ds: loompy.LoomConnection = None, labels: np.array = None, out_file:str = None) -> None:
    names = np.array(["-"])
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
    if ax is None:
        ax = plt.gca()
    if labels is not None or  "DoubletFinderFlag" in ds.ca:
        if  labels is not None:
            names,labels = np.unique(labels,return_inverse=True)
        elif "DoubletFinderFlag" in ds.ca:
            names,labels = np.unique(ds.ca.DoubletFinderFlag,return_inverse=True)
        colors = colorize(names)
        cells = np.random.permutation(labels.shape[0])
        ax.scatter(xy[cells, 0], xy[cells, 1], c=colors[labels][cells], lw=0, s=10)
        h = lambda c: plt.Line2D([], [], color=c, ls="", marker="o")
        ax.legend(handles=[h(colors[i]) for i in range(len(names))],labels=list(names),loc='lower left',
        markerscale=1,frameon=False,fontsize=10)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c='lightgrey', lw=0, s=10)

    sp = ax.set_title("Doublets Flag")
    if out_file is not None:
        plt.savefig(out_file, dpi=144)
    return(sp)
  
    

def doublets_umis(ax: plt.axes = None,ds: loompy.LoomConnection  = None, labels: np.array = None, out_file: str = None) -> None:
    
    if ax is None:
        ax = plt.gca()
    
    if labels is not None:
        doublets = labels
    elif "DoubletFinderFlag" in ds.ca:
        doublets = ds.ca.DoubletFinderFlag
    else:
        doublets = np.array(["(unknown)"] * ds.shape[1])
    umis = [ds.ca.TotalUMI[doublets==0],ds.ca.TotalUMI[doublets>0]]
    pos = [0,1]
    
    res= stats.mannwhitneyu(ds.ca.TotalUMI[doublets>0],ds.ca.TotalUMI[doublets==0],alternative='greater')
    ax.set_title(f'UMI counts (Mann-Whitney pval:'+'{:0.2e}'.format(res.pvalue)+')')
    ax.set_ylabel('UMI counts')
    box_colors = ['r', 'royalblue']
    ax.set_xticklabels(['Singlets','Doublets'], rotation=45, fontsize=8)
    bp = ax.boxplot(umis, positions=pos, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    if out_file is not None:
        plt.savefig(out_file, dpi=144)
    return(bp)
    

def doublets_ngenes(ax: plt.axes = None, ds: loompy.LoomConnection = None, labels: np.array = None, out_file: str= None) -> None:
    
    if ax is None:
        ax = plt.gca()
    if labels is not None:
        doublets = labels
    elif "DoubletFinderFlag" in ds.ca:
        doublets = ds.ca.DoubletFinderFlag
    else:
        doublets = np.array(["(unknown)"] * ds.shape[1])
    ngenes = [ds.ca.NGenes[doublets==0],ds.ca.NGenes[doublets>0]]
    pos = [0,1]
    
    res= stats.mannwhitneyu(ds.ca.NGenes[doublets>0],ds.ca.NGenes[doublets==0],alternative='greater')
    ax.set_title(f' Number of genes (Mann-Whitney pval:'+'{:0.2e}'.format(res.pvalue)+')')
    ax.set_ylabel('Number of genes')
    box_colors = ['r', 'royalblue']
    ax.set_xticklabels(['Singlets','Doublets'], rotation=45, fontsize=8)
    bp = ax.boxplot(ngenes, positions=pos, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    if out_file is not None:
        plt.savefig(out_file, dpi=144)
    return(bp)
    

def fake_doublets_dist(ax: plt.axes = None,doublet_score_A: np.array=None,logprob:np.array=None, xx: np.array=None, score1: float = 1, score2: float=1,score: float=1, out_file: str=None) -> None:
    #fig, ax = plt.subplots(figsize=(12, 12))
    if ax is None:
        ax = plt.gca()
    ax.fill_between(xx.T[0], np.exp(logprob), alpha=0.5)
    ax.plot(doublet_score_A, np.full_like(doublet_score_A, -0.01), '|k', markeredgewidth=1)
    ax.set_ylim(-0.02, 5)
    
    ax.set_title('Fake Doublets distribution (Selected TH: '+str(score)+')')
    if score1 != 1:
        ax.axvline(x=score1, c='r')
    if score2 != 1:
        ax.axvline(x=score2,linestyle='--',c = 'r')
    if score != 1:
        ax.axvline(x=score,linestyle=':',c = 'r')
    ax.set_ylabel('# cells')
    ax.set_xlabel('DoubletFinder score')
    hd = ax.hist(doublet_score_A, bins=30,density=True)
    if out_file is not None:
        plt.savefig(out_file,dpi=144)
    return(hd)    
    