import scipy.sparse as sparse
import numpy as np
import os
from typing import *


class InfomapResult:
	def __init__(self, path: str) -> None:
		with open(path, "r", encoding="utf8") as f:
			line = f.readline()
			line = f.readline()
			line = f.readline()
			labels = []
			flow = []
			rank = []
			indices = []

			ix = 0
			while line:
				items = line.split()
				nodes = items[0].split(":")
				labels.append([int(x) for x in nodes[:-1]])
				rank.append(int(nodes[-1]))
				flow.append(float(items[1]))
				indices.append(int(items[-1]))
				line = f.readline()
				if line.startswith("*"):
					break

			ordering = np.argsort(indices)
			self.labels = np.array(labels)[ordering]
			self.flow = np.array(flow)[ordering]
			self.rank = np.array(rank)[ordering]
			if "undirected" in line:
				self.directed = False
			else:
				self.directed = True
			line = f.readline()  # Eat a comment line
			line = f.readline()  # Now we're on the "*Links root" line

			self.modules: List[np.ndarray] = []
			while line:
				items = line.split()
				n_edges = int(items[3])
				n_modules = int(items[4])
				m = np.zeros((n_modules, n_modules))
				for _ in range(n_edges):
					line = f.readline()
					items = line.split()
					m[int(items[0]) - 1, int(items[1]) - 1] = float(items[2])
				self.modules.append(m)
				line = f.readline()  # Now we're on the next "*Links root" line
