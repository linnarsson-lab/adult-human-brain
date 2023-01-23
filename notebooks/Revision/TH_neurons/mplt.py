import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.patches import Wedge,Circle,Ellipse
import itertools
import matplotlib.markers as mks
from matplotlib.cm import ScalarMappable


from matplotlib import transforms

import loompy
import os
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
import color
colors75,default_102,default_20,default_28 = color.colors75,color.godsnot_102,color.default_20,color.default_28

from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

import pandas as pd

from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex,to_rgba


from typing import Union, List, Sequence, Tuple, Collection, Optional, Callable
from matplotlib.figure import SubplotParams as sppars, Figure
from matplotlib import rcParams, ticker, gridspec, axes

from typing import Optional, Any, List
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
grey_viridis = (0.93359375, 0.93359375, 0.9375, 1)
viridis_grey  = (239/256, 239/256, 240/256,1)
import matplotlib
lightgrey = matplotlib.cm.colors.to_rgba('lightgrey')

def ixs_thatsort_a2b(a: np.ndarray, b: np.ndarray, check_content: bool=True) -> np.ndarray:
    "This is super duper magic sauce to make the order of one list to be like another"
    if check_content:
        assert len(np.intersect1d(a, b)) == len(a), f"The two arrays are not matching"
    return np.argsort(a)[np.argsort(np.argsort(b))]

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



def factors(attr: np.array,
            embed:np.array,cells:bool=None,cmap=None,cmap_source='cytograph',order:np.array=None,
            params:dict=None,fontsize:int= 12,
            annotated:bool=True,title:str=None,legend_heading:str=None,with_legend=True,return_col:bool=False,return_cmap:bool=False, **kwargs) -> None:

    if params is None:
        markerscale = 5
    else:
        markerscale = params['markerscale']

    figsize = kwargs.pop('figsize',(15,8))
    fig = plt.figure(None,figsize)
    # fig = plt.gcf()

    ax = fig.add_axes([0, 0, 0.8, 1])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    
   # ax = plt.subplot(4, 4, 1)
    plt.xticks(())
    plt.yticks(())
    plt.axis("off")
    
    pos = embed
    min_pts = 50
    eps_pct = 60
    nn = NearestNeighbors(n_neighbors=min_pts, algorithm="ball_tree", n_jobs=4)
    nn.fit(pos)
    knn = nn.kneighbors_graph(mode='distance')
    k_radius = knn.max(axis=1).toarray()
    epsilon = (2500 / (pos.max() - pos.min())) * np.percentile(k_radius, eps_pct)
    size = 700000/pos.shape[0]#170000 / pos.shape[0]
    size = kwargs.pop('s',size)

    # if cells is None:
    #     cells_final = np.full(pos.shape[1], True, dtype=bool)
    # else:
    #     cells_final = cells

    if title is not None:
        ax.set_title(title,fontsize=fontsize,style='normal',pad=-100)

    if isinstance(attr[0], str)  or isinstance(attr[0],np.bool_):    

        if isinstance(attr[0],np.bool_):    
            attr = np.array([str(i) for i in attr])
            cmap_source = 'bool_col'

        
        plots=[]
        tag1_names=[]
        
        if order is None:
            attr_list = np.unique(attr)
        else:
            attr_list = np.unique(attr)
            ixs = ixs_thatsort_a2b(attr_list,order)
            attr_list = attr_list[ixs]
        
        col_list=[]
        if cmap_source=='cytograph':
            if cells is not None:
                plt.scatter(x=pos[:, 0][~cells], y=pos[:, 1][~cells], color =grey_viridis, marker='.', lw=0, alpha=1, s = size,rasterized=True,**kwargs)
                for i in range(0,len(attr_list)):
                    loc = np.isin(attr, attr_list[i]).nonzero()[0]
                    cell_b00l = cells[loc]
                    if cmap is None:
                        c = colors75[np.mod(i, 75)]
                        plots.append(plt.scatter(x=pos[loc, 0][cell_b00l], y=pos[loc, 1][cell_b00l], c=[c], marker='.', lw=0, s = size,rasterized=True,**kwargs))
                    else:
                        ctype = attr_list[i]
                        plots.append(plt.scatter(x=pos[loc, 0][cell_b00l], y=pos[loc, 1][cell_b00l], color=cmap[ctype], marker='.', lw=0, alpha=0.8,s = size,rasterized=True,**kwargs))

                    if return_col==True:
                        col_list.append(c)

                    txt = attr_list[i]
                    tag1_names.append(f"{txt}")

            else:
                for i in range(0,len(attr_list)):
                    loc = np.isin(attr, attr_list[i]).nonzero()[0]

                    if cmap is None:
                        if attr_list[i] =='':
                            c = lightgrey#np.array([245/255,	245/255,245/255])
                        else:
                            c = colors75[np.mod(i, 75)]
                        plots.append(plt.scatter(x=pos[loc, 0], y=pos[loc, 1], c=[c], marker='.', lw=0, s = size,rasterized=True,**kwargs))
                    else:
                        ctype = attr_list[i]
                        plots.append(plt.scatter(x=pos[loc, 0], y=pos[loc, 1], color=cmap[ctype], marker='.', lw=0, alpha=0.8,s = size,rasterized=True,**kwargs))

                    if return_col==True:
                        col_list.append(c)

                    txt = attr_list[i]
                    tag1_names.append(f"{txt}")

        elif cmap_source=='bool_col':
            cmap = {'True': np.array([231/255, 109/255, 137/255]), #https://www.flatuicolorpicker.com/colors/deep-blush/
                        'False': lightgrey}# np.array([245/255,	245/255,245/255])}
            
            for i in range(0,len(attr_list)):
                loc = np.isin(attr, attr_list[i]).nonzero()[0]
                ctype = attr_list[i]
                c = cmap[ctype]
                plots.append(plt.scatter(x=pos[loc, 0], y=pos[loc, 1], color=[c], marker='.', lw=0, alpha=0.8,s = size,rasterized=True,**kwargs))

                if return_col==True:
                    col_list.append(c)

                txt = attr_list[i]
                tag1_names.append(f"{txt}")

        elif cmap_source=='scanpy':

            if len(attr_list) <= 20:
                palette = default_20
            elif len(attr_list) <= 28:
                palette = default_28
            elif len(attr_list) <= len(default_102):  # 103 colors
                palette = default_102

            # cmap = plt.get_cmap(palette[:len(attr_list)])
            # colors_list = [to_hex(x) for x in cmap(np.linspace(0, 1, len(attr_list)))]

            colors_list = [to_rgba(x) for x in palette[:len(attr_list)]]

            cmap = dict(zip(attr_list,colors_list))

            for i in range(0,len(attr_list)):
                loc = np.isin(attr, attr_list[i]).nonzero()[0]
                ctype = attr_list[i]
                c = cmap[ctype]
                plots.append(plt.scatter(x=pos[loc, 0], y=pos[loc, 1], color=[c], marker='.', lw=0, alpha=0.8,s = size,rasterized=True,**kwargs))

                if return_col==True:
                    col_list.append(c)

                txt = attr_list[i]
                tag1_names.append(f"{txt}")
        
        if annotated:
            for lbl in range(0,len(attr_list)):
                txt = str(attr_list[lbl])
                if np.sum(attr == attr_list[lbl]) == 0:
                    continue
                (x, y) = np.median(pos[np.where(attr == attr_list[lbl])[0]], axis=0)
                ax.text(x, y, txt, fontsize=12, bbox=dict(facecolor='white', alpha=0.5, ec='none'))

        # if title is not None:
        #     ax.set_title(title,fontsize=12,loc='right')
        
        ax.autoscale_view()


        # plt.colorbar(cax, ax=ax, pad=0.01, fraction=0.08, aspect=30)
        # width = max([len(i) for i in np.unique(tag1_names)])

        height = len(attr_list)*0.042
        if with_legend:
            ax2 = fig.add_axes([0.81, 0, 1, 1])
            ax2.axis("off")

            tag1_name_ =[]
            for i in  tag1_names:
                names = i.split('-')
                names = '\n'.join(names) 
                tag1_name_.append(names)
                    
            if legend_heading is None:
                ax2.legend(plots, tag1_name_, scatterpoints=1, markerscale=markerscale, loc='center', mode='expand', framealpha=0, fontsize=12)
            else:
                legend =ax2.legend(plots, tag1_name_, scatterpoints=1, markerscale=markerscale, loc='center', mode='expand', framealpha=0, fontsize=12,
                title=legend_heading,title_fontsize=16,borderpad=0.1)
                legend._legend_box.align='left'

            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.grid(False)
    
    else:    
        
        if cmap is None:
            cmap = "viridis"
        else:
            cmap = cmap
        
        # sorting the values to plot
        to_color_attr = attr
        order = np.argsort(-to_color_attr,kind='stable')[::-1]
        to_color_attr = to_color_attr[order]
        data_points_x = pos[:,0][order]
        data_points_y = pos[:,1][order]

        vmin,vmax = kwargs.pop('vmin',np.min(to_color_attr)),kwargs.pop('vmax',np.max(to_color_attr))
        print(vmin,vmax)
        normalize = Normalize(vmin = vmin, vmax = vmax)
        print(size)
        if cells is not None:
                print(np.min(to_color_attr[cells]),np.max(to_color_attr[cells]))
                plt.scatter(x=pos[:, 0][~cells], y=pos[:, 1][~cells], c='whitesmoke', marker='.', lw=0, alpha=0.8, s = size,rasterized=True,**kwargs)
                cax = ax.scatter(
                data_points_x[cells],
                data_points_y[cells],
                marker=".",
                c=to_color_attr[cells],
                rasterized=False,
                norm=normalize,
                alpha=0.7,
                s = size,
                cmap = cmap)
        else:
            

            cax = ax.scatter(
                    data_points_x,
                    data_points_y,
                    marker=".",
                    c=to_color_attr,
                    norm=normalize,
                    lw=0, alpha=0.8,
                    s = size,rasterized=True,
                    cmap = cmap,**kwargs
                )
            sm = ScalarMappable(norm = normalize,cmap=cmap)
        # print(vmin,vmax)
# 
        # ax.scatter(x=data_points_x, y=data_points_y, c=to_color_attr, marker='.', s=size, norm=normalize,lw=0,plotnonfinite=True,cmap=cmap)
        if with_legend:

            ax2 = fig.add_axes([0.81, 0, 1, 1])
            ax2.axis("off")

            for i in ['right','left','top','bottom']:
                ax2.spines[i].set_visible(False)
            
            ax2.grid(False)

            cticks = np.linspace(vmin,np.round(vmax),3)
            cticks_fin = kwargs.pop('cticks',cticks)
            cbar_fract = kwargs.pop('cbar_fraction',0.01)
            cbar_aspect = kwargs.pop('cbar_aspect',30)
            cbar_shrink = kwargs.pop('cbar_shrink',1.0)
            
            sm.set_array([])
            cb = plt.colorbar(sm, fraction=cbar_fract,shrink=cbar_shrink, aspect=cbar_aspect,ticks=cticks_fin,
                            location='right')
            cb.ax.tick_params(labelsize=5,pad=0.1,length=2)
            cb.ax.tick_params(axis='y', direction='out')


            for t in cb.ax.get_yticklabels():
                t.set_fontsize(5)
            cb.outline.set_linewidth(0.2)
            cb.dividers.set_linewidth(0.2)
            cb.ax.tick_params(width=0.2)
        ax.axis("off")
        plt.close()
        if return_cmap:
            colorbar = dict({'sm':sm,
                            'vmin':vmin,
                            'vmax':vmax})
            return fig,colorbar


   
            
    ax.axis("off")
    plt.close()

    if return_col==True:
        col_list = dict(zip(attr_list,col_list))
        return fig,col_list
    
    
    return fig


