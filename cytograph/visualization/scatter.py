from typing import Optional, Any, List
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from .colors import Colorizer
from ..utils import div0


def _draw_edges(ax: plt.Axes, pos: np.ndarray, g: np.ndarray, gcolor: str, galpha: float, glinewidths: float) -> None:
	lc = LineCollection(zip(pos[g.row], pos[g.col]), linewidths=0.25, zorder=0, color=gcolor, alpha=galpha)
	ax.add_collection(lc)


def scatterc(xy: np.ndarray, *, c: np.ndarray, colors = None, labels = None, legend: Optional[str] = "outside", g: np.ndarray = None, gcolor: str = "thistle", galpha: float = 0.1, glinewidths: float = 0.25, **kwargs) -> None:
	if colors is None:
		colorizer = Colorizer("colors75").fit(c)
	elif isinstance(colors, str):
		colorizer = Colorizer(colors).fit(c)
	else:
		colorizer = colors

	n_cells = xy.shape[0]
	fig = plt.gcf()
	area = np.prod(fig.get_size_inches())
	marker_size = 100_000 / n_cells * (area / 25)

	ordering = np.random.permutation(xy.shape[0])
	c = c[ordering]
	xy = xy[ordering, :]
	s = kwargs.pop("s", marker_size)
	lw = kwargs.pop("lw", 0)
	plt.scatter(xy[:, 0], xy[:, 1], c=colorizer.transform(c), s=s, lw=lw, **kwargs)
	ax = plt.gca()
	if legend not in [None, False]:
		if labels is None:
			labels = np.unique(c)
		hidden_lines = [Line2D([0], [0], color=clr, lw=4) for clr in colorizer.transform(labels)]
		if legend == "outside":
			ax.legend(hidden_lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
		else:
			ax.legend(hidden_lines, labels, loc=legend)
	if g is not None:
		_draw_edges(ax, xy, g, gcolor, galpha, glinewidths)


def scattern(xy: np.ndarray, *, c: np.ndarray, cmap: Any = "inferno_r", bgval: Any = None, g: np.ndarray = None, gcolor: str = "thistle", galpha: float = 0.1, glinewidths: float = 0.25, **kwargs) -> None:
	n_cells = xy.shape[0]
	fig = plt.gcf()
	area = np.prod(fig.get_size_inches())
	marker_size = 100_000 / n_cells * (area / 25)

	ordering = np.random.permutation(xy.shape[0])
	color = c[ordering]
	xy = xy[ordering, :]
	s = kwargs.pop("s", marker_size)
	lw = kwargs.pop("lw", 0)
	if cmap is not None:
		if isinstance(cmap, str):  # Try to make a Colorizer cmap
			try:
				cmap = Colorizer(cmap).cmap
			except ValueError:
				pass
	cmap = kwargs.pop("cmap", cmap)
	if bgval is not None:
		cells = color != bgval
		plt.scatter(xy[:, 0], xy[:, 1], c="lightgrey", s=s, lw=lw, cmap=cmap, **kwargs)
		plt.scatter(xy[cells, 0], xy[cells, 1], c=color[cells], s=s, lw=lw, cmap=cmap, **kwargs)
	else:
		plt.scatter(xy[:, 0], xy[:, 1], c=color, s=s, lw=lw, cmap=cmap, **kwargs)
	if g is not None:
		ax = plt.gca()
		_draw_edges(ax, xy, g, gcolor, galpha, glinewidths)


def scatterm(xy: np.ndarray, *, c: List[np.ndarray], cmaps: List[Any], bgval: Any = None, labels = None, legend = "outside", max_percentile: float = 98, g: np.ndarray = None, gcolor: str = "thistle", galpha: float = 0.1, glinewidths: float = 0.25, **kwargs) -> None:
	n_cells = xy.shape[0]
	fig = plt.gcf()
	area = np.prod(fig.get_size_inches())
	marker_size = 100_000 / n_cells * (area / 25)

	ordering = np.random.permutation(n_cells)
	c = np.array(c)[:, ordering]
	xy = xy[ordering, :]
	
	winners = np.argmax(c, axis=0)
	c = np.clip(div0(c.T, np.percentile(c, max_percentile, axis=1)).T, 0, 1)
	colors = np.max(c, axis=0)

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
						final_cmaps.append(mcolors.LinearSegmentedColormap.from_list(name=cmap,colors=["white", cmap]))
					else:
						raise ValueError("Unknown color or colormap " + cmap)
		else:
			final_cmaps.append(cmap)

	data = np.zeros((n_cells, 4))
	for i in range(n_cells):
		if bgval is not None and colors[i] == bgval:
			data[i] = (0.8, 0.8, 0.8, 1)
		else:
			data[i] = final_cmaps[winners[i]](colors[i])

	s = kwargs.pop("s", marker_size)
	lw = kwargs.pop("lw", 0)

	plt.scatter(xy[:, 0], xy[:, 1], c=data, s=s, lw=lw, **kwargs)
	if g is not None:
		ax = plt.gca()
		_draw_edges(ax, xy, g, gcolor, galpha, glinewidths)
	
	ax = plt.gca()
	if legend not in [None, False]:
		legend_colors = [cmap(0.8) for cmap in final_cmaps]
		hidden_lines = [Line2D([0], [0], color=clr, lw=4) for clr in legend_colors]
		if legend == "outside":
			ax.legend(hidden_lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))
		else:
			ax.legend(hidden_lines, labels, loc=legend)