from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder



from typing import Mapping, Sequence
from matplotlib import cm, colors

color_alphabet = np.array([
	[240, 163, 255], [0, 117, 220], [153, 63, 0], [76, 0, 92], [0, 92, 49], [43, 206, 72], [255, 204, 153], [128, 128, 128], [148, 255, 181], [143, 124, 0], [157, 204, 0], [194, 0, 136], [0, 51, 128], [255, 164, 5], [255, 168, 187], [66, 102, 0], [255, 0, 16], [94, 241, 242], [0, 153, 143], [224, 255, 102], [116, 10, 255], [153, 0, 0], [255, 255, 128], [255, 255, 0], [255, 80, 5]
]) / 256

colors75 = np.concatenate([color_alphabet, 1 - (1 - color_alphabet) / 2, color_alphabet / 2])


def colorize(x: np.ndarray, *, bgval: Any = None) -> np.ndarray:
	le = LabelEncoder().fit(x)
	xt = le.transform(x)
	colors = colors75[np.mod(xt, 75), :]
	if bgval is not None:
		colors[x == bgval, :] = np.array([0.8, 0.8, 0.8])
	return colors


tube_color_dict = {
	"Bakerloo": "#B36305",
	"Central": "#E32017",
	"Circle": "#FFD300",
	"District": "#00782A",
	"Hammersmith and City": "#F3A9BB",
	"Jubilee": "#A0A5A9",
	"Metropolitan": "#9B0056",
	"Northern": "#000000",
	"Piccadilly": "#003688",
	"Victoria": "#0098D4",
	"Waterloo and City": "#95CDBA",
	"DLR": "#00A4A7",
	"Overground": "#EE7C0E",
	"Tramlink": "#84B817",
	"Cable Car": "#E21836",
	"Crossrail": "#7156A5"
}


tube_colors = np.array([c for c in tube_color_dict.values()])


"""Color palettes in addition to matplotlib's palettes."""

# Colorblindness adjusted vega_10
# See https://github.com/theislab/scanpy/issues/387
vega_10 = list(map(colors.to_hex, cm.tab10.colors))
vega_10_scanpy = vega_10.copy()
vega_10_scanpy[2] = '#279e68'  # green
vega_10_scanpy[4] = '#aa40fc'  # purple
vega_10_scanpy[8] = '#b5bd61'  # kakhi

# default matplotlib 2.0 palette
# see 'category20' on https://github.com/vega/vega/wiki/Scales#scale-range-literals
vega_20 = list(map(colors.to_hex, cm.tab20.colors))

# reorderd, some removed, some added
vega_20_scanpy = [
    # dark without grey:
    *vega_20[0:14:2],
    *vega_20[16::2],
    # light without grey:
    *vega_20[1:15:2],
    *vega_20[17::2],
    # manual additions:
    '#ad494a',
    '#8c6d31',
]
vega_20_scanpy[2] = vega_10_scanpy[2]
vega_20_scanpy[4] = vega_10_scanpy[4]
vega_20_scanpy[7] = vega_10_scanpy[8]  # kakhi shifted by missing grey
# TODO: also replace pale colors if necessary

default_20 = vega_20_scanpy

# https://graphicdesign.stackexchange.com/questions/3682/where-can-i-find-a-large-palette-set-of-contrasting-colors-for-coloring-many-d
# update 1
# orig reference http://epub.wu.ac.at/1692/1/document.pdf
zeileis_28 = [
    "#023fa5",
    "#7d87b9",
    "#bec1d4",
    "#d6bcc0",
    "#bb7784",
    "#8e063b",
    "#4a6fe3",
    "#8595e1",
    "#b5bbe3",
    "#e6afb9",
    "#e07b91",
    "#d33f6a",
    "#11c638",
    "#8dd593",
    "#c6dec7",
    "#ead3c6",
    "#f0b98d",
    "#ef9708",
    "#0fcfc0",
    "#9cded6",
    "#d5eae7",
    "#f3e1eb",
    "#f6c4e1",
    "#f79cd4",
    # these last ones were added:
    '#7f7f7f',
    "#c7c7c7",
    "#1CE6FF",
    "#336600",
]

default_28 = zeileis_28

# from http://godsnotwheregodsnot.blogspot.de/2012/09/color-distribution-methodology.html
godsnot_102 = np.array([
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
])

default_102 = godsnot_102




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