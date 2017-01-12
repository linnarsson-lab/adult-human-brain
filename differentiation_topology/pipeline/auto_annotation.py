import os
from typing import *
import logging
from shutil import copyfile
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import networkx as nx
import loompy
import differentiation_topology as dt
import luigi
from palettable.tableau import Tableau_20
from differentiation_topology import MarkerEnrichment


class AutoAnnotation(luigi.Task):
	"""
	Luigi Task to perform annotation on a Loom file:
		- Auto-annotation
		- Plot
	"""
	build_folder = luigi.Parameter(default="")
	aa_root = luigi.Parameter(default="")
	name = luigi.Parameter()

	def output(self) -> luigi.LocalTarget:
		return luigi.LocalTarget(os.path.join(self.build_folder, "%s_aa.tab" % self.name))

	def requires(self) -> List[Any]:
		return [MarkerEnrichment(build_dir=self.build_folder, name=self.name)]

	def run(self) -> None:
		logging.info("Auto-annotating " + self.filename)
		ds = loompy.connect(self.filename)
		labels = ds.col_attrs["Clusters"]
		aa = dt.AutoAnnotator(ds, root=self.aa_root)
		trinary_prob = np.loadtxt(self.requires()[1].fn())
		(tags, annotations) = aa.annotate(ds, trinary_prob)
		sizes = np.bincount(labels + 1)

		with open(os.path.join(self.build_folder, self.name + "_aa.tab"), "w") as f:
			f.write("\t")
			for j in range(annotations.shape[1]):
				f.write(str(j + 1) + " (" + str(sizes[j]) + ")\t")
			f.write("\n")
			for ix, tag in enumerate(tags):
				f.write(str(tag) + "\t")
				for j in range(annotations.shape[1]):
					f.write(str(annotations[ix, j]) + "\t")
				f.write("\n")

		logging.info("Plotting clusters on graph")
		edges = ds.get_edges("MKNN", axis=1)
		knn = sparse.coo_matrix((edges[2], (edges[0], edges[1])), shape=(ds.shape[1], ds.shape[1]))
		fig = plt.figure(figsize=(10, 10))
		g = nx.from_scipy_sparse_matrix(knn)
		pos = np.array((ds.col_attrs["_tSNE_X"], ds.col_attrs["_tSNE_Y"])).T
		ax = fig.add_subplot(111)

		# Draw the KNN graph first, with gray transparent edges
		plt.title(self.name, fontsize=14, fontweight='bold')
		nx.draw_networkx_edges(g, pos=pos, alpha=0.1, width=0.1, edge_color='gray')

		# Then draw the nodes, colored by label
		block_colors = (np.array(Tableau_20.colors) / 255)[np.mod(labels, 20)]
		nx.draw_networkx_nodes(g, pos=pos, node_color=block_colors, node_size=10, alpha=0.5, linewidths=0)
		for lbl in range(0, max(labels) + 1):
			if np.sum(labels == lbl) == 0:
				continue
			(x, y) = np.median(pos[np.where(labels == lbl)[0]], axis=0)
			text_labels = ["#" + str(lbl + 1)]
			for ix, a in enumerate(annotations[:, lbl]):
				if a >= 0.5:
					text_labels.append(tags[ix].abbreviation)
			text = "\n".join(text_labels)
			ax.text(x, y, text, fontsize=6, bbox=dict(facecolor='gray', alpha=0.3, ec='none'))
		fig.savefig(os.path.join(self.build_folder, self.name + ".png"))
		plt.close()

