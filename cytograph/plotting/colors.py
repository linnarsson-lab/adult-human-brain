from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder

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
