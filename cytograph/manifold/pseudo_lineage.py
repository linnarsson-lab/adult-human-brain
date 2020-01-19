import logging
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import numpy as np
from matplotlib.patches import Polygon
from scipy.interpolate import interp1d

import loompy
from cytograph.pipeline import Cytograph, load_config
from cytograph.interpolation import CatmullRomSpline


def commonest(x: np.ndarray) -> Any:
	unique, pos = np.unique(x, return_inverse=True)
	counts = np.bincount(pos)
	maxpos = counts.argmax()
	return unique[maxpos]


def ribbon(points: np.ndarray, widths: np.ndarray, color: Any, **kwargs: Any) -> Polygon:
	widths = np.array(widths)
	points = np.array(points)
	x = np.concatenate([points[:, 0], points[:, 0][::-1]])
	y = np.concatenate([points[:, 1] + widths / 2, (points[:, 1] - widths / 2)[::-1]])
	outline = np.vstack([x, y]).T
	return Polygon(outline, color=color, **kwargs)


def age_to_num(s: str) -> float:
	s = s.replace("_", ".")
	if s[0] == "e":
		return float(s[1:])
	return float(s[1:]) + 18


class PseudoLineage:
	def __init__(self, slices_pct: int = 5) -> None:
		self.slice_pct = slices_pct

	def fit(self, ds: loompy.LoomConnection) -> None:
		logging.info("Computing pseudoage")
		ages = np.array([age_to_num(x) for x in ds.ca.Age])
		knn = ds.col_graphs.KNN
		k = knn.nnz / knn.shape[0]
		ds.ca.PseudoAge = (knn.astype("bool") @ ages) / k

		logging.info("Slicing pseudoage")
		slice_names: List[str] = []
		with TemporaryDirectory() as tempfolder:
			slices = np.percentile(ds.ca.PseudoAge, np.arange(0, 101, 5))
			logging.info("Collecting cells")
			for (ix, _, view) in ds.scan(axis=1):
				for i in range(len(slices) - 2):
					s1 = slices[i]
					s2 = slices[i + 2]
					slice_name = f"Age{s1:05.2f}to{s2:05.2f}".replace(".", "") + ".loom"
					if slice_name not in slice_names:
						slice_names.append(slice_name)
					cells = ((view.ca.PseudoAge >= s1) & (view.ca.PseudoAge < s2))
					if cells.sum() == 0:
						continue
					fname = os.path.join(tempfolder, slice_name)
					if not os.path.exists(fname):
						with loompy.new(fname) as dsout:
							dsout.add_columns(view.layers[:, cells], col_attrs=view.ca[cells], row_attrs=view.ra)
					else:
						with loompy.connect(fname) as dsout:
							dsout.add_columns(view.layers[:, cells], col_attrs=view.ca[cells], row_attrs=view.ra)

			for slice_name in slice_names:
				fname = os.path.join(tempfolder, slice_name)
				logging.info("Cytograph on " + slice_name)
				with loompy.connect(fname) as ds:
					Cytograph(config=load_config()).fit(ds)
			
			# Use dynamic programming to find the deepest tree (forest), as given by total number of cells along each branch
			logging.info("Computing pseudolineage")
			clusters = "Clusters"
			min_pct = 0.1

			# List of matrices giving the bipartite graph between each pair of layers, weighted by number of shared cells
			overlaps = []
			n_nodes = []  # List of number of nodes (clusters) in each layer
			n_cells = []  # List of arrays giving the number of cells in each cluster
			n_layers = len(slice_names)

			# Compute the bipartite graphs between layers
			for t in range(n_layers):
				# Link clusters from layer t to clusters from layer t + 1
				logging.info(f"{slice_names[t]}.loom")
				with loompy.connect(os.path.join(tempfolder, slice_names[t])) as ds1:
					n_nodes.append(ds1.ca[clusters].max() + 1)
					n_cells.append(np.zeros(n_nodes[t]))
					for c in range(n_nodes[t]):
						n_cells[t][c] = (ds1.ca[clusters] == c).sum()
					if t >= n_layers - 1:
						break
					with loompy.connect(os.path.join(tempfolder, slice_names[t + 1])) as ds2:
						overlap = np.zeros((np.unique(ds1.ca[clusters]).shape[0], np.unique(ds2.ca[clusters]).shape[0]), dtype="int")
						for i in np.unique(ds1.ca[clusters]):
							cells1 = ds1.ca.CellID[ds1.ca[clusters] == i]
							for j in np.unique(ds2.ca[clusters]):
								cells2 = ds2.ca.CellID[ds2.ca[clusters] == j]
								overlap[i, j] = np.intersect1d(cells1, cells2).shape[0]
						overlaps.append(overlap)

			# List of arrays keeping track of the depth of the deepest tree starting at each node in the layer
			# Depth defined as sum of the number of shared cells along the branch
			depths = [np.zeros(n, dtype="int") for n in n_nodes]
			edges = [np.zeros(n, dtype="int") for n in n_nodes[1:]]  # List of arrays giving the predecessor of each cluster (or -1 if no predecessor)
			for t in range(0, n_layers - 1):
				for i in range(n_nodes[t + 1]):
					# Now find the widest deepest branch from any node j in layer t to node i in layer t + 1
					# Widest, deepest meaning: greatest sum of depth up to node j in layer t plus number of shared cells
					# But disallowing any branch with less than min_pct % shared cells
					best_j = -1
					best_depth = 0
					for j in range(n_nodes[t]):
						pct_overlapping = 100 * overlaps[t][j, i] / (n_cells[t][j] + n_cells[t + 1][i])
						if pct_overlapping > min_pct:
							depth = depths[t][j] + overlaps[t][j, i]
							if depth > best_depth:
								best_depth = depth
								best_j = j
					edges[t][i] = best_j

			# Now we have
			#
			# edges:    List of arrays giving the index of the predecessor of each cluster (or -1 if no predecessor exists)
			# overlaps: List of matrices giving the number of cells shared between clusters in layer t and t + 1
			# n_nodes:  List of number of nodes (clusters) in each layer
			# n_cells:  List of arrays of number of cells in each node (cluster)

			# Now position the nodes of each layer such that no edges cross
			ypositions = [np.arange(n_nodes[0])]
			for t in range(len(edges)):
				pos = np.full(n_nodes[t + 1], -1)
				for i in range(pos.shape[0]):
					prev = edges[t][i]
					if(prev) >= 0:
						pos[i] = ypositions[t][prev]
				ordering = np.argsort(pos)
				mapping = dict(zip(ordering, range(len(ordering))))
				ypositions.append(np.array([mapping[i] for i in range(len(ordering))]))
			# Make the positions proportional to the number of cells (cumulative)
			max_pos = 0
			for i, pos in enumerate(ypositions):
				with loompy.connect(os.path.join(tempfolder, slice_names[i])) as ds0:
					n_clusters = ds0.ca[clusters].max() + 1
					ncells = np.array([(ds0.ca[clusters] == i).sum() for i in range(n_clusters)])
					total = 0
					new_pos = np.zeros_like(pos)
					for j in range(len(pos)):
						cluster = np.where(pos == j)[0]
						new_pos[cluster] = total + ncells[cluster] / 2
						total += ncells[cluster]
				ypositions[i] = new_pos / 1000
				max_pos = max(max_pos, max(ypositions[i]))

			for i, pos in enumerate(ypositions):
				ypositions[i] += (max_pos - np.max(pos)) / 2

			# Then position the layers properly in time
			xpositions = []
			for i in range(n_layers):
				with loompy.connect(os.path.join(tempfolder, slice_names[i])) as ds0:
					xpositions.append(np.mean(ds0.ca.PseudoAge))
			
			# Now project each individual cell to the pseudolineage
			logging.info("Projecting cells to pseudolineage")
			cell_to_xy = {}
			for t in range(len(n_nodes) - 1):
				with loompy.connect(os.path.join(tempfolder, slice_names[t])) as ds0:
					with loompy.connect(os.path.join(tempfolder, slice_names[t + 1])) as ds1:
						for i in range(n_nodes[t + 1]):
							if edges[t][i] != -1:
								y1 = ypositions[t][edges[t][i]]
								y2 = ypositions[t + 1][i]
								offset = (xpositions[t + 1] - xpositions[t]) / 4
								overlapping_cells = (ds1.ca[clusters] == i) & (ds1.ca.PseudoAge < slices[t + 2])
								crs = np.array(CatmullRomSpline(n_points=100).fit_transform(np.array([[slices[t + 1] - offset, y1], [slices[t + 1], y1], [slices[t + 2], y2], [slices[t + 2] + offset, y2]])))
								widths = np.linspace(n_cells[t][edges[t][i]], n_cells[t + 1][i], num=100) / 1500
								f = interp1d(crs[:, 0], crs[:, 1])
								fw = interp1d(crs[:, 0], widths)
								y = f(ds1.ca.PseudoAge[overlapping_cells]) + np.random.normal(scale=fw(ds1.ca.PseudoAge[overlapping_cells]) / 6, size=overlapping_cells.sum())
								for i, ix in enumerate(np.where(overlapping_cells)[0]):
									cell_to_xy[ds1.ca.CellID[ix]] = [ds1.ca.PseudoAge[ix], y[i]]
						# Draw the leftmost pseudoage slice
						if t == 0:
							for i in range(n_nodes[0]):
								y1 = ypositions[0][i]
								y2 = ypositions[0][i]
								widths = np.linspace(n_cells[t][i], n_cells[t][i], num=100) / 1500
								overlapping_cells = (ds0.ca[clusters] == i) & (ds0.ca.PseudoAge < slices[1])
								y = y1 + np.random.normal(scale=widths[0] / 6, size=overlapping_cells.sum())
								for i, ix in enumerate(np.where(overlapping_cells)[0]):
									cell_to_xy[ds1.ca.CellID[ix]] = [ds0.ca.PseudoAge[ix], y[i]]
						# Draw the rightmost pseudoage slice
						if t == len(n_nodes) - 2:
							for i in range(n_nodes[-1]):
								y1 = ypositions[t][edges[t][i]]
								y2 = ypositions[t + 1][i]
								widths = np.linspace(n_cells[t][edges[t][i]], n_cells[t + 1][i], num=100) / 1500
								overlapping_cells = (ds1.ca[clusters] == i) & (ds1.ca.PseudoAge > slices[-2])
								y = y2 + np.random.normal(scale=widths[-1] / 6, size=overlapping_cells.sum())
								for i, ix in enumerate(np.where(overlapping_cells)[0]):
									cell_to_xy[ds1.ca.CellID[ix]] = [ds1.ca.PseudoAge[ix], y[i]]

			logging.info("Saving pseudolineage projection back in original file")
			xy = np.zeros_like(ds.ca.TSNE)
			for i, cellid in enumerate(cell_to_xy.keys()):
				j = np.where(ds.ca.CellID == cellid)[0][0]
				xy[j] = cell_to_xy[cellid]
			ds.ca.PseudoLineage = xy
