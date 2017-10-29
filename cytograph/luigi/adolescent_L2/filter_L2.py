from typing import *
import os
import csv
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
import numpy_groupies.aggregate_numpy as npg
import scipy.stats
import tempfile


class FilterL2(luigi.Task):
	"""
	Level 2 filtering of bad clusters
	"""
	major_class = luigi.Parameter()
	tissue = luigi.Parameter(default="All")

	def requires(self) -> luigi.Task:
		return cg.ClusterL2(major_class=self.major_class, tissue=self.tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L2_" + self.major_class + "_" + self.tissue + ".filtered.loom"))

	def run(self) -> None:
		logging = cg.logging(self)
		logging.info("Filtering bad clusters")
		dsout = None  # type: loompy.LoomConnection
		accessions = None  # type: np.ndarray
		with self.output().temporary_path() as out_file:
			ds = loompy.connect(self.input().fn)
			if not len(set(ds.Clusters)) == ds.Clusters.max() + 1:
				raise ValueError("There are holes in the cluster ID sequence!")
			labels = ds.Clusters
			n_labels = len(set(labels))
			remove = []

			# Remove outliers
			if (ds.Outliers == 1).sum() > 0:
				remove.append(ds.Clusters[ds.Outliers == 1][0])
				logging.info("Removing outliers")

			# Remove clusters that lack enriched genes
			logging.info("Checking for clusters with no enriched genes")
			logging.info("Trinarizing")
			trinaries = cg.Trinarizer().fit(ds)
			logging.info("Computing cluster gene enrichment scores")
			(markers, enrichment, qvals) = cg.MarkerSelection(10).fit(ds)
			data = trinaries[markers, :].T
			for ix in range(n_labels):
				total_score = data[ix, ix * 10:(ix + 1) * 10].sum()
				if total_score < 2:
					remove.append(ix)
					logging.info(f"Cluster {ix} score: {total_score:.1f} < 2 (removing).")
				else:
					logging.info(f"Cluster {ix} score: {total_score:.1f}")

			# Remove clusters that express genes of the wrong major class
			logging.info("Checking for clusters with markers of wrong major class")
			nix_genes = {
				"Neurons": ["Stmn2"],
				"Oligos": ['Mog', 'Mobp', 'Neu4'],
				"Vascular": ['Cldn5', 'Fn1', 'Acta2'],
				"Immune": ['Ctss', 'Lyz2', 'Cx3cr1', 'Pf4'],
				"Astrocytes": ['Aqp4'],
				"Ependymal": ["Foxj1", "Ttr"],
				"Blood": ['Hbb-bt', 'Hbb-bh1', 'Hbb-bh2', 'Hbb-y', 'Hbb-bs', 'Hba-a1', 'Hba-a2', 'Hba-x'],
				"PeripheralGlia": ["Mpz", "Prx", 'Col20a1', 'Ifi27l2a'],
			}
			for lbl in range(n_labels):
				# Clusters with markers of other major class
				n_cells_in_cluster = (ds.Clusters == lbl).sum()
				for cls in nix_genes.keys():
					if cls == self.major_class:
						continue
					for gene in nix_genes[cls]:
						if gene not in ds.Gene:
							logging.warn("Couldn't use '" + gene + "' to nix clusters")
						gix = np.where(ds.Gene == gene)[0][0]
						if np.count_nonzero(ds[gix, :][ds.Clusters == lbl]) > 0.5 * n_cells_in_cluster:
							# But let it slide if this marker is abundant in the whole tissue
							if np.count_nonzero(ds[gix, :]) < 0.25 * ds.shape[1]:
								logging.info("Nixing cluster {} because {} was detected".format(lbl, gene))
								remove.append(lbl)

			retain = np.sort(np.setdiff1d(np.arange(n_labels), remove))
			temp: List[int] = []
			for i in retain:
				temp += list(np.where(ds.Clusters == i)[0])
			cells = np.sort(np.array(temp))

			# Renumber the clusters
			d = dict(zip(retain, np.arange(len(set(retain)) + 1)))
			new_clusters = np.array([d[x] if x in d else -1 for x in ds.Clusters])
			logging.info(f"Keeping {cells.shape[0]} of {ds.shape[1]} cells")
			for (ix, selection, vals) in ds.batch_scan(cells=cells, axis=1, batch_size=cg.memory().axis1):
				ca = {k: v[selection] for k, v in ds.col_attrs.items()}
				ca["Clusters"] = new_clusters[selection]
				if dsout is None:
					dsout = loompy.create(out_file, vals, ds.row_attrs, ca)
				else:
					dsout.add_columns(vals, ca)

			# Filter the KNN and MKNN edges
			(a, b, w) = ds.get_edges("KNN", axis=1)
			mask = np.logical_and(np.in1d(a, cells), np.in1d(b, cells))
			a = a[mask]
			b = b[mask]
			w = w[mask]
			d = dict(zip(np.sort(cells), np.arange(cells.shape[0])))
			a = np.array([d[x] for x in a])
			b = np.array([d[x] for x in b])
			dsout.set_edges("KNN", a, b, w, axis=1)

			(a, b, w) = ds.get_edges("MKNN", axis=1)
			mask = np.logical_and(np.in1d(a, cells), np.in1d(b, cells))
			a = a[mask]
			b = b[mask]
			w = w[mask]
			d = dict(zip(np.sort(cells), np.arange(cells.shape[0])))
			a = np.array([d[x] for x in a])
			b = np.array([d[x] for x in b])
			dsout.set_edges("MKNN", a, b, w, axis=1)

			logging.info(f"Filtering {n_labels} -> {len(set(new_clusters[cells]))} clusters")
