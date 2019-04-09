from typing import *
import logging
from collections import defaultdict
import numpy as np
import numpy_groupies as npg
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, leaves_list
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker as ticker
import loompy
import os
import logging as lg
import luigi
import random
import string
from sklearn.preprocessing import LabelEncoder


def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	""" ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide(a, b)
		c[~np.isfinite(c)] = 0  # -inf inf NaN
	return c


def cap_select(labels: np.ndarray, items: np.ndarray, max_n: int) -> np.ndarray:
	"""
	Return a list of items but with no more than max_n entries
	having each unique label
	"""
	n_labels = np.max(labels) + 1
	sizes = np.bincount(labels, minlength=n_labels)
	result = []  # type: List[int]
	for lbl in range(n_labels):
		n = min(max_n, sizes[lbl])
		selected = np.where(labels == lbl)[0]
		result = result + list(np.random.choice(selected, n, False))
	return items[np.array(result)]
