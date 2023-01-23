from typing import Optional, Any, List
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from typing import Union, List, Sequence, Tuple, Collection, Optional, Callable
from matplotlib.figure import SubplotParams as sppars, Figure
from matplotlib import rcParams, ticker, gridspec, axes

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable


grey_viridis = (0.93359375, 0.93359375, 0.9375, 1)


class Colorizer:
	def __init__(self, colors: str = "alphabet") -> None:
		if colors == "alphabet":
			self.cmap = colors75
		elif colors == "tube":
			self.cmap = tube_colors
		else:
			raise ValueError("Colors must be 'alphabet' or 'tube'")

	def fit(self, x: np.ndarray) -> "Colorizer":
		self.encoder = LabelEncoder().fit(x)
		return self

	def transform(self, x: np.ndarray, *, bgval: Any = None) -> np.ndarray:
		if bgval is not None:
			xt = x.copy()
			xt[x != bgval] = self.encoder.transform(x[x != bgval])
			xt[x == bgval] = 0
		else:
			xt = self.encoder.transform(x)
		colors = self.cmap[np.mod(xt, self.cmap.shape[0]), :]
		if bgval is not None:
			colors[x == bgval, :] = np.array([0.8, 0.8, 0.8])
		return colors

	def fit_transform(self, x: np.ndarray, *, bgval: Any = None) -> np.ndarray:
		self.fit(x)
		return self.transform(x, bgval=bgval)

def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c

def draw_colorbar(colorbar,ax_size:list=[0.81, 0, 1, 1],title:str=None,yticklabels=None,xticklabels=None,**kwargs):

    colormappable,vmin,vmax = colorbar['sm'],colorbar['vmin'],colorbar['vmax']
    fig = plt.gcf()
    ax2 = fig.add_axes(ax_size)
    # ax2.axis("off")

    for i in ['right','left','top','bottom']:
        ax2.spines[i].set_visible(False)
    
    ax2.grid(False)

    cticks = np.linspace(vmin,np.round(vmax),3)
    cticks_fin = kwargs.pop('cticks',cticks)
    cbar_fract = kwargs.pop('cbar_fraction',0.01)
    cbar_aspect = kwargs.pop('cbar_aspect',30)
    cbar_shrink = kwargs.pop('cbar_shrink',1.0)
    sm = colormappable
    sm.set_array([])
    cb = plt.colorbar(sm, fraction=cbar_fract,shrink=cbar_shrink, aspect=cbar_aspect,ticks=cticks_fin,cax=ax2,
                    **kwargs)
    cb.ax.tick_params(labelsize=5,pad=0.1,length=2)
    cb.ax.tick_params(axis='y', direction='out')

    if yticklabels is not None:
        cb.ax.set_yticklabels(yticklabels)
    if xticklabels is not None:
        cb.ax.set_xticklabels(xticklabels)

    for t in cb.ax.get_yticklabels():
        t.set_fontsize(5)
    cb.outline.set_linewidth(0.2)
    cb.dividers.set_linewidth(0.2)
    cb.ax.tick_params(width=0.2)

    if title is not None:
        cb.ax.set_title(title,fontsize=9,style='normal',pad=-20)

    return fig
            

def draw_legend(final_cmaps,labels,ax_size,fontsize:int=7,both=False,**kwargs):
    width=max([len(l) for l in labels])
    height=len(labels)*0.5
    
    fig = plt.gcf()

    ms = kwargs.pop("ms", 1)
    alpha = kwargs.pop("alpha", 1)
    markerstyle = kwargs.pop("marker", "o")
    lw = 0

    ax = fig.add_axes(ax_size)
    try:
        legend_colors = [cmap(0.8) for cmap in final_cmaps]
    except TypeError:
        legend_colors = final_cmaps
   
    hidden_lines = [Line2D([0], [0], color=clr, ls='', marker=markerstyle,alpha=alpha,lw=lw) for clr in legend_colors]
  
    labelspacing =  kwargs.pop("labelspacing", 0.5)
    cb = plt.legend(hidden_lines, labels, frameon=False,markerscale=ms,
    fontsize=fontsize, loc = 'center left',handletextpad=0.01,labelspacing=labelspacing)
    plt.axis('off')

    return ax

def make_grid_spec(
    ax_or_figsize: Union[Tuple[int, int]],
    nrows: int,
    ncols: int,
    wspace: Optional[float] = None,
    hspace: Optional[float] = None,
    width_ratios: Optional[Sequence[float]] = None,
    height_ratios: Optional[Sequence[float]] = None,
) -> Tuple[Figure, gridspec.GridSpecBase]:
    kw = dict(
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    if isinstance(ax_or_figsize, tuple):
        fig = pl.figure(figsize=ax_or_figsize)
        return fig, gridspec.GridSpec(nrows, ncols, **kw)
    else:
        ax = ax_or_figsize
        ax.axis('off')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax.figure, ax.get_subplotspec().subgridspec(nrows, ncols, **kw)

def scatterm(xy: np.ndarray, *, c: List[np.ndarray], cmaps: List[Any], ax=None,fig=None,bgval: Any = None, labels = None, legend = "outside", max_percentile: float = 98, g: np.ndarray = None, gcolor: str = "thistle", galpha: float = 0.1, glinewidths: float = 0.25, **kwargs) -> None:
    n_cells = xy.shape[0]
    
    figsize = kwargs.pop('figsize',(8,8))
    if ax is None:  
        fig = plt.figure(None,figsize)  
        ax = fig.add_axes([0, 0, 0.8, 1])

    area = np.prod(fig.get_size_inches())
    marker_size = 100_000 / n_cells * (area / 25)

  #  ordering = np.random.permutation(n_cells)
   # c = np.array(c)[:, ordering]
    #xy = xy[ordering, :]
	
    winners = np.argmax(c, axis=0)
    max_val = np.percentile(c, max_percentile, axis=1)
    assert np.all(max_val > 0), f"{max_percentile}th percentile is zero (increase max_percentile to fix)"

    c = np.clip(div0(c.T, max_val).T, 0, 1)
    colors = np.max(c, axis=0)

    final_cmaps = []
    for cmap in cmaps:
        if isinstance(cmap, str):
            if '#' in cmap:
                final_cmaps.append(mcolors.LinearSegmentedColormap.from_list(name=cmap, colors=["white", cmap]))
            else:
                try:
                    final_cmaps.append(Colorizer(cmap).cmap)
                except ValueError:
                    try:
                        final_cmaps.append(plt.cm.get_cmap(cmap))
                    except ValueError:
                        if cmap in mcolors.BASE_COLORS or cmap in mcolors.TABLEAU_COLORS or cmap in mcolors.CSS4_COLORS:
                            final_cmaps.append(mcolors.LinearSegmentedColormap.from_list(name=cmap, colors=["white", cmap]))
                        else:
                            raise ValueError("Unknown color or colormap " + cmap)
        else:
            final_cmaps.append(cmap)

    data = np.zeros((n_cells, 4))
    for i in range(n_cells):
        if bgval is not None and colors[i] == bgval:
            data[i] = grey_viridis
        else:
            data[i] = final_cmaps[winners[i]](colors[i])

    s = kwargs.pop("s", marker_size)
    lw = kwargs.pop("lw", 0)

    if bgval is not None:
        bgpoints = colors == bgval
        ax.scatter(xy[bgpoints, 0], xy[bgpoints, 1], color='lightgrey', s=s, lw=lw ,**kwargs)
        ax.scatter(xy[~bgpoints, 0], xy[~bgpoints, 1], c=data[~bgpoints], s=s, lw=lw, **kwargs)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c=data, s=s, lw=lw, **kwargs)
    if g is not None:
        ax_ = plt.gca()
        _draw_edges(ax_, xy, g, gcolor, galpha, glinewidths)

    ax_ = plt.gca()
    if legend not in [None, False]:
        legend_colors = [cmap(0.8) for cmap in final_cmaps]
        hidden_lines = [Line2D([0], [0], color=clr, lw=4) for clr in legend_colors]
        if legend == "outside":
            ax_.legend(hidden_lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax_.legend(hidden_lines, labels, loc=legend)
            
            
            
    return final_cmaps


        
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

def scatterm_prodonly(xy: np.ndarray, *, c: List[np.ndarray], cmaps: List[Any],bgval: Any = None,\
     nowhite:bool=False,reverse:bool=False,labels = None, legend = "outside", \
        max_percentile: float = 98, g: np.ndarray = None, gcolor: str = "thistle",\
             galpha: float = 0.1, glinewidths: float = 0.25, **kwargs) -> None:
    n_cells = xy.shape[0]

    figsize = kwargs.pop('figsize',(8,8))
    fig = plt.figure(None,figsize)  
    ax = fig.add_axes([0, 0, 0.8, 1])
    area = np.prod(fig.get_size_inches())
    marker_size = 100_000 / n_cells * (area / 25)


    c = np.array(c)
    xy = xy
    c = np.clip(div0(c.T, np.percentile(c, max_percentile, axis=1)).T, 0, 1)
    colors = np.prod(c, axis=0)
    

    final_cmaps = []
    for cmap in cmaps:
        if isinstance(cmap, str):
            try:
                final_cmaps.append(Colorizer(cmap).cmap)
            except ValueError:
                try:
                    final_cmaps.append(plt.cm.get_cmap(cmap))
                except ValueError:
                    if cmap in mcolors.BASE_COLORS or cmap in mcolors.TABLEAU_COLORS or cmap in mcolors.CSS4_COLORS:
                        if nowhite:
                            tmp = mcolors.LinearSegmentedColormap.from_list(name=cmap, colors=["white", cmap])
                            newcolour = tmp(np.linspace(0,1,256))
                            newcolour = newcolour[20:,:]
                            if reverse:
                                newcolour=newcolour[::-1]
                            final_cmaps.append(mcolors.ListedColormap(newcolour))
                        else:
                            final_cmaps.append(mcolors.LinearSegmentedColormap.from_list(name=cmap, colors=["white", cmap]))
                    else:
                        raise ValueError("Unknown color or colormap " + cmap)
        else:
            final_cmaps.append(cmap)
    
#     cmap_prod = plt.cm.get_cmap(cmap_prod)

    data = np.zeros((n_cells, 4))
    pos = np.zeros(n_cells)
    for i in range(n_cells):
        if (bgval is not None) and (colors[i] == bgval):
                data[i] = (0.8, 0.8, 0.8, 1)
        else:
            data[i] = final_cmaps[0](colors[i])
            pos[i] = 1
    bgpoints = colors == bgval
    pos =  (~bgpoints).nonzero()[0]
    s = kwargs.pop("s", marker_size)
    lw = kwargs.pop("lw", 0)
    
    if bgval is not None:
        bgpoints = colors == bgval
        ax.scatter(xy[bgpoints, 0], xy[bgpoints, 1], c="lightgrey", s=s, lw=lw, **kwargs)
        ax.scatter(xy[~bgpoints, 0], xy[~bgpoints, 1], c=data[~bgpoints], s=s, lw=lw, **kwargs)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c=data, s=s, lw=lw, **kwargs)
    if g is not None:
        ax = plt.gca()
        _draw_edges(ax, xy, g, gcolor, galpha, glinewidths)
   
    ax.axis('off') 
    
    return data,pos,final_cmaps



def scattern(xy: np.ndarray, *, c: np.ndarray, cmap: Any = "inferno_r",ax=None,fig=None, bgval: Any = None, g: np.ndarray = None, gcolor: str = "thistle", galpha: float = 0.1, glinewidths: float = 0.25, **kwargs) -> None:
    n_cells = xy.shape[0]
    
    figsize = kwargs.pop('figsize',(8,8))
    if ax is None:
        fig = plt.figure(None,figsize)  
        ax = fig.add_axes([0, 0, 0.8, 1])

    area = np.prod(fig.get_size_inches())
    marker_size = 100_000 / n_cells * (area / 25)

    # ordering = np.random.permutation(xy.shape[0])
    color = c#[ordering]
    xy = xy#[ordering, :]
    s = kwargs.pop("s", marker_size)
    lw = kwargs.pop("lw", 0)
    if cmap is not None:
        if isinstance(cmap, str):
            try:
                cmap = Colorizer(cmap).cmap
            except ValueError:
                try:
                    cmap = plt.cm.get_cmap(cmap)
                except ValueError:
                    if cmap in mcolors.BASE_COLORS or cmap in mcolors.TABLEAU_COLORS or cmap in mcolors.CSS4_COLORS:
                        cmap = mcolors.LinearSegmentedColormap.from_list(name=cmap, colors=["white", cmap])
                    else:
                        raise ValueError("Unknown color or colormap " + cmap)
    cmap = kwargs.pop("cmap", cmap)
    if bgval is not None:
        cells = color != bgval
        ax.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=s, lw=lw, cmap=cmap, **kwargs)
        ax.scatter(xy[cells, 0], xy[cells, 1], c=color[cells], s=s, lw=lw, cmap=cmap, **kwargs)
    else:
        plt.scatter(xy[:, 0], xy[:, 1], c=color, s=s, lw=lw, cmap=cmap, **kwargs)
    if g is not None:
        ax = plt.gca()
        _draw_edges(ax, xy, g, gcolor, galpha, glinewidths)

    return cmap


     