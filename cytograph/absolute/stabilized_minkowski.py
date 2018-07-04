import numpy as np
import numba
from typing import *


@numba.jit("float32(float64[:], float64[:])", nopython=True, cache=True)
def stabilized_minkowski(x: np.ndarray, y: np.ndarray) -> float:
	p = 10
	scale = np.sum(x) / np.sum(y)  # Would be slightly better to do a proper regression here
	y_scaled = y * scale
	x_adj = (np.sqrt(x) + 0.8 * np.sqrt(x + 1)) / 1.8
	y_adj = (np.sqrt(y_scaled) + 0.8 * np.sqrt(y_scaled + 1)) / 1.8

	return np.sum((x_adj - y_adj) ** p) ** (1 / p)
