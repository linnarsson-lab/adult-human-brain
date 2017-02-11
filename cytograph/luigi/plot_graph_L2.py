from typing import *
import os
import logging
import loompy
import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import luigi
from palettable.tableau import Tableau_20


class PlotGraphL2(luigi.Task):
	"""
	Luigi Task to plot the MKNN graph, level 2
	"""
	project = luigi.Parameter(default="Adolescent")
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> List[luigi.Task]:
		return [cg.ClusterLayoutL2(tissue=self.tissue, major_class=self.major_class, project=self.project), cg.AutoAnnotateL2(tissue=self.tissue, major_class=self.major_class, project=self.project)]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join("loom_builds", self.major_class + "_" + self.tissue + ".mknn.pdf"))

	def run(self) -> None:
		logging.info("Plotting MKNN graph")
		# Parse the auto-annotation tags
		tags = []
		with open(self.input()[1].fn, "r") as f:
			content = f.readlines()[1:]
			for line in content:
				tags.append(line.split('\t')[1].replace(",", "\n"))
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input()[0].fn)
			n_cells = ds.shape[1]
			valid = ds.col_attrs["_Valid"].astype('bool')
			(a, b, w) = ds.get_edges("MKNN", axis=1)
			mknn = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells)).tocsr()[valid, :][:, valid]
			sfdp = np.vstack((ds.col_attrs["_SFDP_X"], ds.col_attrs["_SFDP_Y"])).transpose()[valid, :]
			labels = ds.col_attrs["Clusters"][valid]

			fig = plt.figure(figsize=(10, 10))
			g = nx.from_scipy_sparse_matrix(mknn)
			ax = fig.add_subplot(111)
			plt.title(self.major_class + " (" + self.tissue + ")", fontsize=14, fontweight='bold')

			# Draw the KNN graph first, with gray transparent edges
			nx.draw_networkx_edges(g, pos=sfdp, alpha=0.1, width=0.1, edge_color='gray')
			# Then draw the nodes, colored by label
			block_colors = (np.array(Tableau_20.colors) / 255)[np.mod(labels, 20)]
			nx.draw_networkx_nodes(g, pos=sfdp, node_color=block_colors, node_size=10, alpha=0.5, linewidths=0)

			mg_pos = []
			for lbl in range(0, max(labels) + 1):
				if np.sum(labels == lbl) == 0:
					continue
				(x, y) = np.median(sfdp[np.where(labels == lbl)[0]], axis=0)
				mg_pos.append((x, y))
				text = "#" + str(lbl)
				if len(tags[lbl]) > 0:
					text += "\n" + tags[lbl]
				ax.text(x, y, text, fontsize=6, bbox=dict(facecolor='gray', alpha=0.3, ec='none'))

			fig.savefig(out_file, format="pdf")
			plt.close()
