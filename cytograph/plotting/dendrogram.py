import numpy as np
from matplotlib.collections import LineCollection


def dendrogram(z: np.ndarray, *, leaf_positions: np.ndarray = None, orientation: str = "top") -> LineCollection:
	"""
	Create a dendrogram, in the form of a matplotlib LineCollection, in data space, and optionally
	with unevenly spaced leaves given by leaf_positions.
	"""

	lines = []
	n = z.shape[0] + 1
	xpos = np.zeros(z.shape[0])
	if leaf_positions is None:
		leaf_positions = np.arange(n)
	else:
		leaf_positions = np.sort(leaf_positions)

	# From scipy docs about the z matrix (they call it Z):
	# A (n - 1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1]
	# are combined to form cluster (n + i). A cluster with an index less than n corresponds to one of the n
	# original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The
	# fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.

	for i in range(z.shape[0]):
		# Draw a Π-shaped glyph with the left leg extending to the top of cluster z[i, 0] and the right leg
		# to cluster z[i, 1] and the crossbar at level z[i, 2].
		if z[i, 0] < n:
			left_y = 0
			left_x = leaf_positions[int(z[i, 0])]
		else:
			left_y = z[int(z[i, 0]) - n, 2]
			left_x = xpos[int(z[i, 0]) - n]
		if z[i, 1] < n:
			right_y = 0
			right_x = leaf_positions[int(z[i, 1])]
		else:
			right_y = z[int(z[i, 1]) - n, 2]
			right_x = xpos[int(z[i, 1]) - n]
		xpos[i] = left_x + (right_x - left_x) / 2
		crossbar_y = z[i, 2]
		if orientation == "top":
			lines.append([[left_x, left_y], [left_x, crossbar_y], [right_x, crossbar_y], [right_x, right_y]])
		else:
			raise NotImplementedError("Only 'top' orientation is supported")
	return LineCollection(lines, colors="black")
