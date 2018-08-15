import numpy as np
import numba
from typing import *


@numba.jit("float32(float64[:], float64[:], float64, float64)", nopython=True, cache=True)
def stabilized_minkowski(x: np.ndarray, y: np.ndarray, n: float = 5000, p: float = 10) -> float:
	x_scaled = x * n / np.sum(x)
	y_scaled = y * n / np.sum(y)
	x_adj = (np.sqrt(x_scaled) + 0.8 * np.sqrt(x_scaled + 1)) / 1.8
	y_adj = (np.sqrt(y_scaled) + 0.8 * np.sqrt(y_scaled + 1)) / 1.8
	return np.sum((x_adj - y_adj) ** p) ** (1 / p)


@numba.jit("float32(float64[:], float64[:])", nopython=True, cache=True)
def minkowski10(x: np.ndarray, y: np.ndarray) -> float:
	p = 10
	return np.sum((x - y) ** p) ** (1 / p)
