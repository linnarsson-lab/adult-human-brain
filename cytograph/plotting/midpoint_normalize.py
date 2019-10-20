import numpy as np
from matplotlib.colors import Normalize


class MidpointNormalize(Normalize):
	def __init__(self, vmin: float = None, vmax: float = None, midpoint: float = None, clip: bool = False) -> None:
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value: float, clip: bool = None) -> np.ndarray:
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))
