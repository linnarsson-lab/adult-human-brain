import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import loompy
from ..embedding import tsne
from .colors import colorize
from sklearn.manifold import TSNE


def doublets_TSNE(ds: loompy.LoomConnection, out_file: str, labels: np.array = None) -> None:
    
    if 'HPF' in ds.ca:
        xy = tsne(ds.ca.HPF)
    elif 'PCA' in ds.ca:
        angle=0.5
        perplexity=30
        verbose=False
        xy = TSNE(angle=angle, perplexity=perplexity, verbose=verbose).fit_transform(ds.ca.PCA)
    ds.ca.TSNE = xy
    plt.figure(figsize=(12, 12))
    if labels is not None:
        labels = labels
    elif "DoubletFinderFlag" in ds.ca:
        labels = ds.ca.DoubletFinderFlag
    else:
        labels = np.array(["(unknown)"] * ds.shape[1])

    for lbl in np.unique(labels):
        cells = labels == lbl
        plt.scatter(xy[:, 0][cells], xy[:, 1][cells], c=colorize(labels)[cells], label=lbl, lw=0, s=10)
    plt.legend()
    plt.title("Doublets")

    plt.savefig(out_file, dpi=144)
    plt.close()

def doublets_umis(ds: loompy.LoomConnection, out_file: str, labels: np.array = None) -> None:
    
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.canvas.set_window_title('UMI counts Doublets')
    
    if labels is not None:
        doublets = labels
    elif "DoubletFinderFlag" in ds.ca:
        doublets = ds.ca.DoubletFinderFlag
    else:
        doublets = np.array(["(unknown)"] * ds.shape[1])
    umis = [ds.ca.TotalUMI[doublets==0],ds.ca.TotalUMI[doublets==1]]
    pos = [0,1]
    bp = ax.boxplot(umis, positions=pos)
    res= stats.mannwhitneyu(ds.ca.TotalUMI[doublets==1],ds.ca.TotalUMI[doublets==0],alternative='greater')
    ax.set_title(f'Comparison of UMI counts in doublet vs. singlet prediction (Mann-Whitney pval:'+'{:0.3e}'.format(res.pvalue)+')')
    ax.set_ylabel('UMI counts')
    box_colors = ['r', 'royalblue']
    ax.set_xticklabels(['Singlets','Doublets'], rotation=45, fontsize=8)
    plt.savefig(out_file, dpi=144)
    plt.close()
def doublets_ngenes(ds: loompy.LoomConnection, out_file: str, labels: np.array = None) -> None:
    
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.canvas.set_window_title('Number of genes Doublets')
    
    if labels is not None:
        doublets = labels
    elif "DoubletFinderFlag" in ds.ca:
        doublets = ds.ca.DoubletFinderFlag
    else:
        doublets = np.array(["(unknown)"] * ds.shape[1])
    ngenes = [ds.ca.NGenes[doublets==0],ds.ca.NGenes[doublets==1]]
    pos = [0,1]
    bp = ax.boxplot(ngenes, positions=pos)
    res= stats.mannwhitneyu(ds.ca.NGenes[doublets==1],ds.ca.NGenes[doublets==0],alternative='greater')
    ax.set_title(f'Comparison of number of genes expressed in doublet vs. singlet prediction (Mann-Whitney pval:'+'{:0.3e}'.format(res.pvalue)+')')
    ax.set_ylabel('Number of genes')
    box_colors = ['r', 'royalblue']
    ax.set_xticklabels(['Singlets','Doublets'], rotation=45, fontsize=8)
    plt.savefig(out_file, dpi=144)
    plt.close()

def fake_doublets_dist(doublet_score_A: np.array,logprob:np.array, xx: np.array, score1: float, score2: float,score: float, out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.fill_between(xx.T[0], np.exp(logprob), alpha=0.5)
    plt.plot(doublet_score_A, np.full_like(doublet_score_A, -0.01), '|k', markeredgewidth=1)
    plt.ylim(-0.02, 5)
    plt.hist(doublet_score_A, bins=30,density=True)
    ax.set_title('Fake Doublets distribution (Picked TH: '+str(score)+')')
    plt.axvline(x=score1, c='r')
    plt.axvline(x=score2,linestyle='--',c = 'r')
    ax.set_ylabel('# cells')
    ax.set_xlabel('DoubletFinder score')
    plt.savefig(out_file,dpi=144)
    plt.close()

def nn_dist(nn1: np.array, nn2: np.array,out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    nn = [nn1,nn2]
    plt.hist(nn, bins=30,density=True)
    
    ax.set_title('NN distribution')
  
    plt.savefig(out_file,dpi=144)
    plt.close()