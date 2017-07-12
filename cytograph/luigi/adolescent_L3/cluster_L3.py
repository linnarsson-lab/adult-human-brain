from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
import numpy_groupies.aggregate_numpy as npg
import scipy.stats


class ClusterL3(luigi.Task):
	"""
	Level 3 clustering of the adolescent dataset
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")
	method = luigi.Parameter(default='dbscan')  # or 'hdbscan'
	n_genes = luigi.IntParameter(default=1000)
	gtsne = luigi.BoolParameter(default=True)
	alpha = luigi.FloatParameter(default=1)
	pep = luigi.FloatParameter(default=0.01)

	def requires(self) -> luigi.Task:
		return [
			cg.ClusterL2(tissue=self.tissue, major_class=self.major_class),
			cg.AggregateL2(tissue=self.tissue, major_class=self.major_class)
		]

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L3_" + self.major_class + "_" + self.tissue + ".loom"))
		
	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			dsout = None  # type: loompy.LoomConnection
			accessions = None  # type: np.ndarray
			ds = loompy.connect(self.input()[0].fn)
			logging.info("Extracting clusters from " + self.input()[0].fn)

			# Remove clusters that express genes of the wrong major class
			nix_genes = {
				"Neurons": ["Stmn2"],
				"Oligos": ['Mog', 'Mobp', 'Neu4', 'Bmp4', 'Enpp6', 'Gpr17'],
				"Vascular": ['Cldn5', 'Fn1', 'Acta2'],
				"Immune": ['C1qc', 'Ctss', 'Lyz2', 'Cx3cr1', 'Pf4'],
				"AstroEpendymal": ['Aqp4', 'Gja1', 'Btbd17', 'Cldn10', "Foxj1", "Tmem212", "Ccdc153", "Ak7", "Stoml3", "Ttr", "Folr1", "Cldn2"],
				"Blood": ['Hbb-bt', 'Hbb-bh1', 'Hbb-bh2', 'Hbb-y', 'Hbb-bs', 'Hba-a1', 'Hba-a2', 'Hba-x'],
				"PeripheralGlia": ["Mpz", "Prx", 'Col20a1', 'Ifi27l2a', "Sfrp5"],
			}
			dsagg = loompy.connect(self.input()[1].fn)
			nix_clusters = set()
			for lbl in range(max(ds.col_attrs["Clusters"]) + 1):
				n_cells_in_cluster = (ds.Clusters == lbl).sum()
				if n_cells_in_cluster < 20:
					logging.info("Nixing cluster {} because less than 20 cells".format(lbl))
					nix_clusters.add(lbl)
				else:
					for cls in nix_genes.keys():
						if cls == self.major_class:
							continue
						for gene in nix_genes[cls]:
							if gene not in ds.Gene:
								logging.warn("Couldn't use '" + gene + "' to nix clusters")
							gix = np.where(ds.Gene == gene)[0][0]
							if np.count_nonzero(ds[gix, :][ds.Clusters == lbl]) > 0.5 * n_cells_in_cluster:
								if np.count_nonzero(ds[gix, :]) < 0.25 * ds.shape[1]:
									logging.info("Nixing cluster {} because {} was detected".format(lbl, gene))
									nix_clusters.add(lbl)
			logging.info("Nixing " + str(len(nix_clusters)) + " clusters")
			nix_attr = np.zeros(dsagg.shape[1], dtype='int')
			nix_attr[np.array(list(nix_clusters), dtype='int')] = 1
			nix_attr[dsagg.Outliers == 1] = 1
			dsagg.set_attr("Nixed", nix_attr, axis=1)
			nix_attr = np.zeros(ds.shape[1], dtype='int')
			for lbl in nix_clusters:
				nix_attr[ds.Clusters == lbl] = 1
			nix_attr[ds.Outliers == 1] = 1
			ds.set_attr("Nixed", nix_attr, axis=1)
			cells = np.where(nix_attr == 0)[0]

			logging.info("Keeping " + str(len(cells)) + " cells")
			for (ix, selection, vals) in ds.batch_scan(cells=np.array(cells), axis=1, batch_size=cg.memory().axis1):
				ca = {}
				for key in ds.col_attrs:
					ca[key] = ds.col_attrs[key][selection]
				if dsout is None:
					dsout = loompy.create(out_file, vals, ds.row_attrs, ca)
				else:
					dsout.add_columns(vals, ca)
			dsout.close()

			logging.info("Learning the manifold")
			ds = loompy.connect(out_file)
			ml = cg.ManifoldLearning(self.n_genes, self.gtsne, self.alpha)
			(knn, mknn, tsne) = ml.fit(ds)
			ds.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			ds.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			ds.set_attr("_X", tsne[:, 0], axis=1)
			ds.set_attr("_Y", tsne[:, 1], axis=1)

			# logging.info("Clustering on the manifold")
			# cls = cg.Clustering(method=self.method)
			# labels = cls.fit_predict(ds)
			# ds.set_attr("Clusters", labels, axis=1)
			# n_labels = np.max(labels) + 1

			ds.close()
