from abc import abstractmethod
from typing import Any, Dict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

color_alphabet = np.array([
	[240, 163, 255], [0, 117, 220], [153, 63, 0], [76, 0, 92], [0, 92, 49], [43, 206, 72], [255, 204, 153], [128, 128, 128], [148, 255, 181], [143, 124, 0], [157, 204, 0], [194, 0, 136], [0, 51, 128], [255, 164, 5], [255, 168, 187], [66, 102, 0], [255, 0, 16], [94, 241, 242], [0, 153, 143], [224, 255, 102], [116, 10, 255], [153, 0, 0], [255, 255, 128], [255, 255, 0], [255, 80, 5]
]) / 256

colors75 = np.concatenate([color_alphabet, 1 - (1 - color_alphabet) / 2, color_alphabet / 2])


def colorize(x: np.ndarray, *, bgval: Any = None, cmap: np.ndarray = None) -> np.ndarray:
	le = LabelEncoder().fit(x)
	xt = le.transform(x)
	if cmap is None:
		cmap = colors75
	colors = cmap[np.mod(xt, 75), :]
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


class ColorScheme:
	@abstractmethod
	def fit(self, x: np.ndarray) -> None:
		pass

	@abstractmethod
	def transform(self, x: np.ndarray) -> np.ndarray:
		pass

	def fit_transform(self, x: np.ndarray) -> np.ndarray:
		self.fit(x)
		return self.transform(x)

	@abstractmethod
	def dict(self) -> Dict[str, str]:
		pass


class NamedColorScheme(ColorScheme):
	def __init__(self, names, colors, permute: bool = False):
		self.colors = np.array(colors)
		self.names = np.array(names)
		self.permute = permute
		if self.permute:
			self.colors = np.random.permutation(self.colors)

	def fit(self, x: np.ndarray) -> None:
		pass

	def transform(self, x: np.ndarray) -> np.ndarray:
		assert np.all(np.isin(np.unique(x), self.names)), "Array contains undefined named colors"
		indices = np.nonzero(x[:, None] == self.names)[1]
		return self.colors[indices]

	def dict(self) -> Dict[str, str]:
		return dict(zip(self.names, [matplotlib.colors.to_hex(c) for c in self.colors]))


class DiscreteColorScheme(ColorScheme):
	def __init__(self, colors, permute: bool = False) -> None:
		self.colors = np.array(colors)
		self.permute = permute
		if self.permute:
			self.colors = np.random.permutation(self.colors)
		self.encoding = LabelEncoder()

	def fit(self, x: np.ndarray) -> None:
		self.encoding.fit(x)
	
	def transform(self, x: np.ndarray) -> np.ndarray:
		indices = self.encoding.transform(x) % len(self.colors)
		return self.colors[indices]

	def dict(self) -> Dict[str, str]:
		colors = [matplotlib.colors.to_hex(c) for c in self.colors]
		return dict(zip(colors, colors))


class Colorizer:
	def __init__(self, scheme, permute: bool = False, interpolated: bool = False) -> None:
		self.interpolated = interpolated
		if isinstance(scheme, ColorScheme):
			self.scheme = scheme
		elif scheme == "tube":
			self.scheme = DiscreteColorScheme([
				"#B36305", "#E32017", "#FFD300", "#00782A",
				"#F3A9BB", "#A0A5A9", "#9B0056", "#000000",
				"#003688", "#0098D4", "#95CDBA", "#00A4A7",
				"#EE7C0E", "#84B817", "#E21836", "#7156A5"], permute)
		elif scheme == "colors18":
			self.scheme = DiscreteColorScheme([
				"#51574a", "#447c69", "#74c493",
				"#8e8c6d", "#e4bf80", "#e9d78e",
				"#e2975d", "#f19670", "#e16552",
				"#993767", "#65387d", "#4e2472",
				"#9163b6", "#e279a3", "#e0598b",
				"#7c9fb0", "#5698c4", "#9abf88"
			], permute)
		elif scheme == "colors75":
			self.scheme = DiscreteColorScheme([
				[0.9375, 0.63671875, 0.99609375],
				[0., 0.45703125, 0.859375],
				[0.59765625, 0.24609375, 0.],
				[0.296875, 0., 0.359375],
				[0., 0.359375, 0.19140625],
				[0.16796875, 0.8046875, 0.28125],
				[0.99609375, 0.796875, 0.59765625],
				[0.5, 0.5, 0.5],
				[0.578125, 0.99609375, 0.70703125],
				[0.55859375, 0.484375, 0.],
				[0.61328125, 0.796875, 0.],
				[0.7578125, 0., 0.53125],
				[0., 0.19921875, 0.5],
				[0.99609375, 0.640625, 0.01953125],
				[0.99609375, 0.65625, 0.73046875],
				[0.2578125, 0.3984375, 0.],
				[0.99609375, 0., 0.0625],
				[0.3671875, 0.94140625, 0.9453125],
				[0., 0.59765625, 0.55859375],
				[0.875, 0.99609375, 0.3984375],
				[0.453125, 0.0390625, 0.99609375],
				[0.59765625, 0., 0.],
				[0.99609375, 0.99609375, 0.5],
				[0.99609375, 0.99609375, 0.],
				[0.99609375, 0.3125, 0.01953125],
				[0.96875, 0.81835938, 0.99804688],
				[0.5, 0.72851562, 0.9296875],
				[0.79882812, 0.62304688, 0.5],
				[0.6484375, 0.5, 0.6796875],
				[0.5, 0.6796875, 0.59570312],
				[0.58398438, 0.90234375, 0.640625],
				[0.99804688, 0.8984375, 0.79882812],
				[0.75, 0.75, 0.75],
				[0.7890625, 0.99804688, 0.85351562],
				[0.77929688, 0.7421875, 0.5],
				[0.80664062, 0.8984375, 0.5],
				[0.87890625, 0.5, 0.765625],
				[0.5, 0.59960938, 0.75],
				[0.99804688, 0.8203125, 0.50976562],
				[0.99804688, 0.828125, 0.86523438],
				[0.62890625, 0.69921875, 0.5],
				[0.99804688, 0.5, 0.53125],
				[0.68359375, 0.97070312, 0.97265625],
				[0.5, 0.79882812, 0.77929688],
				[0.9375, 0.99804688, 0.69921875],
				[0.7265625, 0.51953125, 0.99804688],
				[0.79882812, 0.5, 0.5],
				[0.99804688, 0.99804688, 0.75],
				[0.99804688, 0.99804688, 0.5],
				[0.99804688, 0.65625, 0.50976562],
				[0.46875, 0.31835938, 0.49804688],
				[0., 0.22851562, 0.4296875],
				[0.29882812, 0.12304688, 0.],
				[0.1484375, 0., 0.1796875],
				[0., 0.1796875, 0.09570312],
				[0.08398438, 0.40234375, 0.140625],
				[0.49804688, 0.3984375, 0.29882812],
				[0.25, 0.25, 0.25],
				[0.2890625, 0.49804688, 0.35351562],
				[0.27929688, 0.2421875, 0.],
				[0.30664062, 0.3984375, 0.],
				[0.37890625, 0., 0.265625],
				[0., 0.09960938, 0.25],
				[0.49804688, 0.3203125, 0.00976562],
				[0.49804688, 0.328125, 0.36523438],
				[0.12890625, 0.19921875, 0.],
				[0.49804688, 0., 0.03125],
				[0.18359375, 0.47070312, 0.47265625],
				[0., 0.29882812, 0.27929688],
				[0.4375, 0.49804688, 0.19921875],
				[0.2265625, 0.01953125, 0.49804688],
				[0.29882812, 0., 0.],
				[0.49804688, 0.49804688, 0.25],
				[0.49804688, 0.49804688, 0.],
				[0.49804688, 0.15625, 0.00976562]
			], permute)
		elif scheme == "age":
			self.scheme = DiscreteColorScheme([
				[0.43529411, 0.23921568, 0.58039215, 1.],
				[0.26905037, 0.45397924, 0.70349865, 1.],
				[0.45490196, 0.67843137, 0.81960784, 1.],
				[0.66635909, 0.84759708, 0.91188005, 1.],
				[0.87843137, 0.95294118, 0.97254902, 1.],
				[0.99607843, 0.87843137, 0.56470588, 1.],
				[0.99146482, 0.67735486, 0.37808535, 1.],
				[0.95686275, 0.42745098, 0.26274510, 1.],
				[0.83929258, 0.18454441, 0.15286428, 1.]
			], permute)
		elif scheme == "regions":
			self.scheme = NamedColorScheme(
				["Head", "Brain", "Forebrain", "Telencephalon", "Diencephalon", "Midbrain", "Hindbrain", "Pons", "Cerebellum", "Medulla"],
				["#eed8c9", "#a49592", "#f26b38", "#cb4335", "#e9c413", "#45ad78", "#a7226e", "#cc527a", "#589bf2", "#9b59b6"], permute)
		elif scheme == "subregions":
			self.scheme = NamedColorScheme([
				'Brain', 'Caudate+Putamen', 'Cerebellum', 'Cortex', 'Cortex entorhinal', 'Cortex frontal', 'Cortex hemisphere A',
				'Cortex hemisphere B', 'Cortex occipital', 'Cortex parietal', 'Cortex temporal', 'Cortical hem', 'Diencephalon',
				'Forebrain', 'Head', 'Hindbrain', 'Hippocampus', 'Hypothalamus', 'Medulla', 'Midbrain', 'Midbrain dorsal',
				'Midbrain ventral', 'Pons', 'Striatum', 'Subcortex', 'Telencephalon', 'Thalamus'
			], [
				"#a49592", "#db5345", "#589bf2", "#eb6355", "#eb6355", "#eb6355", "#eb6355", "#eb6355", "#eb6355", "#eb6355",
				"#eb6355", "#eb6355", "#e9c413", "#f26b38", "#eed8c9", "#a7226e", "#bb3325", "#c9a403", "#9b59b6", "#45ad78",
				"#359d68", "#55bd88", "#cc527a", "#f34a4a", "#fc9d9a", "#cb4335", "#e1f5c4"
			], permute)
		elif scheme == "classes":
			self.scheme = NamedColorScheme(
				['Neuron', 'Neuroblast', 'Neuronal IPC', 'Radial glia', "Glioblast", "Oligo", "Fibroblast", "Neural crest", "Placodes", "Immune", "Vascular", "Erythrocyte", "Failed"],
				["#5384db", "#5d25c4", "#ab3bc4", "#27b35d", "#447c69", "#9cba19", "#c48351", "#eec79f", "#70510e", "#e7c31f", "#e12e12", "#ff617f", "#9f9f9f"], permute)
		elif scheme == 'sex':
			self.scheme = NamedColorScheme(
							['M', 'F'],
							['#0098d4', '#f3a9bb'], permute)
		else:
			raise ValueError(f"Unrecognized scheme '{scheme}'")

	def transform(self, x: np.ndarray) -> np.ndarray:
		return self.scheme.transform(x)
	
	def fit(self, x: np.ndarray) -> "Colorizer":
		self.scheme.fit(x)
		return self

	def fit_transform(self, x: np.ndarray) -> np.ndarray:
		return self.scheme.fit_transform(x)

	@property
	def cmap(self) -> matplotlib.colors.Colormap:
		if self.interpolated:
			return matplotlib.colors.LinearSegmentedColormap.from_list(name=self.scheme, colors=self.scheme.colors)
		else:
			return matplotlib.colors.ListedColormap(self.scheme.colors)

	def dict(self) -> Dict[str, str]:
		return self.scheme.dict()

	def plot(self, show_labels: bool = True) -> None:
		n_colors = len(self.scheme.colors)
		fig = plt.figure(figsize=(15, 15))
		ax = fig.add_subplot(111)
		ax.set_aspect("equal")
		for ix, color in enumerate(self.scheme.colors):
			start = ix % 10
			y = ix // 10
			rect = matplotlib.patches.Rectangle((start, y), 1, 1, color=color)
			ax.add_patch(rect)
			if show_labels:
				if type(self.scheme) is NamedColorScheme and self.scheme.names is not None:
					plt.text(start + 0.5, y + 0.5, self.scheme.names[ix], ha="center", va="center", fontsize=14)
				else:
					plt.text(start + 0.5, y + 0.5, matplotlib.colors.to_hex(self.scheme.colors[ix]), ha="center", va="center", fontsize=14)
			plt.xlim(0, min(10, n_colors))
			plt.ylim(n_colors // 10 + 1, 0)
		plt.axis("off")
		plt.show()