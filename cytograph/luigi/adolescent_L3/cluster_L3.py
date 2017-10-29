from typing import *
import os
from shutil import copyfile
import csv
#import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
import numpy_groupies.aggregate_numpy as npg
import scipy.stats


params = {  # eps_pct and min_pts
	"L3_SpinalCord_Inhibitory": [50, 10],
	"L3_SpinalCord_Excitatory": [60, 10],
	"L3_Olfactory_Inhibitory": [75, 10],
	"L3_Enteric_Neurons": [60, 10],
	"L3_Sensory_Neurons": [60, 10],
	"L3_Sympathetic_Neurons": [60, 10],	
	"L3_Hypothalamus_Peptidergic": [60, 10],
	"L3_Hindbrain_Inhibitory": [60, 10],
	"L3_Hindbrain_Excitatory": [60, 10],
	"L3_Brain_Neuroblasts": [70, 20],
	"L3_Forebrain_Inhibitory": [75, 10],
	"L3_Forebrain_Excitatory": [75, 10],
	"L3_DiMesencephalon_Inhibitory": [70, 10],
	"L3_DiMesencephalon_Excitatory": [70, 10],
	"L3_Brain_Granule": [80, 70],
	"L3_Brain_CholinergicMonoaminergic": [60, 10],
	"L3_Striatum_MSN": [78, 60]
}


class ClusterL3(luigi.Task):
	"""
	Level 3 clustering of the adolescent dataset
	"""
	target = luigi.Parameter()  # e.g. Forebrain_Excitatory
	n_enriched = luigi.Parameter(default=500)  # Number of enriched genes per cluster to use for manifold learning
	
	def requires(self) -> Iterator[luigi.Task]:
		tissues: List[str] = []
		for tissue, schedule in pooling_schedule_L3.items():
			for aa, sendto in schedule:
				if sendto == self.target:
					if tissue in tissues:
						continue
					yield [cg.FilterL2(tissue=tissue, major_class="Neurons"), cg.AggregateL2(tissue=tissue, major_class="Neurons")]
					tissues.append(tissue)

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "L3_" + self.target + ".loom"))
		
	def run(self) -> None:
		logging = cg.logging(self, True)
		dsout: loompy.LoomConnection = None
		accessions: loompy.LoomConnection = None
		with self.output().temporary_path() as out_file:
			logging.info("Gathering cells for " + self.target)
			enriched_markers: List[np.ndarray] = []  # The enrichment vector for each selected cluster
			cells_found = False
			for in_file, agg_file in self.input():
				tissue = os.path.basename(in_file.fn).split("_")[2].split(".")[0]
				ds = loompy.connect(in_file.fn)
				dsagg = loompy.connect(agg_file.fn)
				enrichment = dsagg.layer["enrichment"][:, :]
				labels = ds.col_attrs["Clusters"]
				ordering: np.ndarray = None
				logging.info(tissue)

				# Figure out which cells should be collected
				cells: List[int] = []
				# clusters_seen: List[int] = []  # Clusters for which there was some schedule
				clusters_seen: Dict[int, str] = {}
				for from_tissue, schedule in pooling_schedule_L3.items():
					if from_tissue != tissue:
						continue

					# Where to send clusters when no rules match
					_default_schedule: str = None
					for aa_tag, sendto in schedule:
						if aa_tag == "*":
							_default_schedule = sendto

					# For each cluster in the tissue
					for ix, agg_aa in enumerate(dsagg.col_attrs["AutoAnnotation"]):
						# For each rule in the schedule
						for aa_tag, sendto in schedule:
							if aa_tag in agg_aa.split(","):
								if ix in clusters_seen:
									logging.info(f"{tissue}/{ix}/{agg_aa}: {aa_tag} -> {sendto} (overruled by '{clusters_seen[ix]}')")
								else:
									clusters_seen[ix] = f"{aa_tag} -> {sendto}"
									if sendto == self.target:
										logging.info(f"## {tissue}/{ix}/{agg_aa}: {aa_tag} -> {sendto}")
										# Keep track of the gene order in the first file
										if accessions is None:
											accessions = ds.row_attrs["Accession"]
										if ordering is None:
											ordering = np.where(ds.row_attrs["Accession"][None, :] == accessions[:, None])[1]
										cells += list(np.where(labels == ix)[0])
										enriched_markers.append(np.argsort(-enrichment[:, ix][ordering]))
									else:
										logging.info(f"{tissue}/{ix}/{agg_aa}: {aa_tag} -> {sendto}")
						if ix not in clusters_seen:
							if _default_schedule is None:
								logging.info(f"{tissue}/{ix}/{agg_aa}: No matching rule")
							else:
								clusters_seen[ix] = f"{aa_tag} -> {sendto}"
								if sendto == self.target:
									logging.info(f"## {tissue}/{ix}/{agg_aa}: {aa_tag} -> {sendto}")
									# Keep track of the gene order in the first file
									if accessions is None:
										accessions = ds.row_attrs["Accession"]
									if ordering is None:
										ordering = np.where(ds.row_attrs["Accession"][None, :] == accessions[:, None])[1]
									cells += list(np.where(labels == ix)[0])
									enriched_markers.append(np.argsort(-enrichment[:, ix][ordering]))
								else:
									logging.info(f"{tissue}/{ix}/{agg_aa}: {aa_tag} -> {sendto}")

				if len(cells) > 0:
					cells = np.sort(np.array(cells))
					cells_found = True
					for (ix, selection, vals) in ds.batch_scan(cells=cells, axis=1, batch_size=cg.memory().axis1):
						ca = {}
						for key in ds.col_attrs:
							ca[key] = ds.col_attrs[key][selection]
						if dsout is None:
							dsout = loompy.create(out_file, vals[ordering, :], ds.row_attrs, ca)
						else:
							dsout.add_columns(vals[ordering, :], ca)

			if not cells_found:
				raise ValueError(f"No cells matched any schedule for {self.target}")

			# Figure out which enriched markers to use
			ix = 0
			temp: List[int] = []
			while len(temp) < self.n_enriched:
				for j in range(len(enriched_markers)):
					if enriched_markers[j][ix] not in temp:
						temp.append(enriched_markers[j][ix])
				ix += 1
			genes = np.sort(np.array(temp))

			logging.info("Learning the manifold")
			ds = loompy.connect(out_file)
			ml = cg.ManifoldLearning2(gtsne=True, alpha=1, genes=genes)
			(knn, mknn, tsne) = ml.fit(ds)
			ds.set_edges("KNN", knn.row, knn.col, knn.data, axis=1)
			ds.set_edges("MKNN", mknn.row, mknn.col, mknn.data, axis=1)
			ds.set_attr("_X", tsne[:, 0], axis=1)
			ds.set_attr("_Y", tsne[:, 1], axis=1)

			logging.info("Clustering on the manifold")
			(eps_pct, min_pts) = (65, 20)
			if "L3_" + self.target in params:
				(eps_pct, min_pts) = params["L3_" + self.target]
			logging.info(f"MKNN-Louvain with eps_pct {eps_pct}, min_pts {min_pts}")
			cls = cg.Clustering(method="mknn_louvain", eps_pct=eps_pct, min_pts=min_pts)
			labels = cls.fit_predict(ds)
			n_labels = len(set(labels))
			ds.set_attr("Clusters", labels, axis=1)
			logging.info(f"Found {labels.max() + 1} clusters")
			distance = 0.2
			if self.target == "Striatum_MSN":
				distance = 0.4
			cg.Merger(min_distance=distance).merge(ds)
			# Merge twice to deal with super-similar clusters like granule cells
			cg.Merger(min_distance=distance).merge(ds)
			n_labels = ds.col_attrs['Clusters'].max() + 1
			logging.info(f"Merged to {n_labels} clusters")

			ds.close()


pooling_schedule_L3 = {
	"Cortex1": [
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "Forebrain_Inhibitory"),
		("DG-GC", "Brain_Granule"),
		("@VGLUT1", "Forebrain_Excitatory"),
		("@VGLUT2", "Forebrain_Excitatory"),
		("@VGLUT3", "Forebrain_Excitatory"),
		("@NBL", "Brain_Neuroblasts")
	],
	"Cortex2": [
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "Forebrain_Inhibitory"),
		("DG-GC", "Brain_Granule"),
		("@VGLUT1", "Forebrain_Excitatory"),
		("@VGLUT2", "Forebrain_Excitatory"),
		("@VGLUT3", "Forebrain_Excitatory"),
		("@NBL", "Brain_Neuroblasts")
	],
	"Cortex3": [
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "Forebrain_Inhibitory"),
		("DG-GC", "Brain_Granule"),
		("@VGLUT1", "Forebrain_Excitatory"),
		("@VGLUT2", "Forebrain_Excitatory"),
		("@VGLUT3", "Forebrain_Excitatory"),
		("@NBL", "Brain_Neuroblasts")
	],
	"Hippocampus": [
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "Forebrain_Inhibitory"),
		("DG-GC", "Brain_Granule"),
		("@NBL", "Brain_Neuroblasts"),
		("@VGLUT1", "Forebrain_Excitatory"),
		("@VGLUT2", "Forebrain_Excitatory"),
		("@VGLUT3", "Forebrain_Excitatory"),
	],
	"StriatumDorsal": [
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "Forebrain_Inhibitory"),
		("DG-GC", "Brain_Granule"),
		("@VGLUT1", "Forebrain_Excitatory"),
		("@VGLUT2", "Forebrain_Excitatory"),
		("@VGLUT3", "Forebrain_Excitatory"),
		("@NBL", "Brain_Neuroblasts")
	],
	"StriatumVentral": [
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "Forebrain_Inhibitory"),
		("DG-GC", "Brain_Granule"),
		("@VGLUT1", "Forebrain_Excitatory"),
		("@VGLUT2", "Forebrain_Excitatory"),
		("@VGLUT3", "Forebrain_Excitatory"),
		("@NBL", "Brain_Neuroblasts")
	],
	"Amygdala": [
		("@GABAGLUT1", "Amygdala_Other"),
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "Forebrain_Inhibitory"),
		("DG-GC", "Brain_Granule"),
		("@VGLUT1", "Forebrain_Excitatory"),
		("@VGLUT2", "Forebrain_Excitatory"),
		("@VGLUT3", "Forebrain_Excitatory"),
		("@NBL", "Brain_Neuroblasts")
	],
	"Olfactory": [
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@NIPC", "Brain_Neuroblasts"),
		("@VGLUT1", "Forebrain_Excitatory"),
		("@VGLUT2", "Forebrain_Excitatory"),
		("@VGLUT3", "Forebrain_Excitatory"),
		("@NBL", "Brain_Neuroblasts"),
		("*", "Olfactory_Inhibitory")
	],
	"Hypothalamus": [
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@OXT", "Hypothalamus_Peptidergic"),
		("@AVP", "Hypothalamus_Peptidergic"),
		("@GNRH", "Hypothalamus_Peptidergic"),
		("@AGRP", "Hypothalamus_Peptidergic"),
		("@HCRT", "Hypothalamus_Peptidergic"),
		("@PMCH", "Hypothalamus_Peptidergic"),
		("@POMC", "Hypothalamus_Peptidergic"),
		("@TRH", "Hypothalamus_Peptidergic"),
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("@NIPC", "Brain_Neuroblasts"),
		("@VGLUT1", "DiMesencephalon_Excitatory"),
		("@VGLUT2", "DiMesencephalon_Excitatory"),
		("@VGLUT3", "DiMesencephalon_Excitatory"),
		("@GABA", "DiMesencephalon_Inhibitory")
	],
	"MidbrainDorsal": [
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@SER", "Brain_CholinergicMonoaminergic"),
		("@DA", "Brain_CholinergicMonoaminergic"),
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "DiMesencephalon_Inhibitory"),
		("@VGLUT1", "DiMesencephalon_Excitatory"),
		("@VGLUT2", "DiMesencephalon_Excitatory"),
		("@VGLUT3", "DiMesencephalon_Excitatory")
	],
	"MidbrainVentral": [
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@SER", "Brain_CholinergicMonoaminergic"),
		("@DA", "Brain_CholinergicMonoaminergic"),
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "DiMesencephalon_Inhibitory"),
		("@VGLUT1", "DiMesencephalon_Excitatory"),
		("@VGLUT2", "DiMesencephalon_Excitatory"),
		("@VGLUT3", "DiMesencephalon_Excitatory")
	],
	"Thalamus": [
		("MSN-D1", "Striatum_MSN"),
		("MSN-D2", "Striatum_MSN"),
		("@SER", "Brain_CholinergicMonoaminergic"),
		("@DA", "Brain_CholinergicMonoaminergic"),
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GABA", "DiMesencephalon_Inhibitory"),
		("@VGLUT1", "DiMesencephalon_Excitatory"),
		("@VGLUT2", "DiMesencephalon_Excitatory"),
		("@VGLUT3", "DiMesencephalon_Excitatory")
	],
	"Cerebellum": [
		("@NIPC", "Brain_Neuroblasts"),
		("CB-PC", "Hindbrain_Inhibitory"),
		("CB-GC", "Brain_Granule"),
		("@GLY", "Hindbrain_Inhibitory"),
		("@GABA", "Hindbrain_Inhibitory"),
		("@VGLUT1", "Hindbrain_Excitatory"),
		("@VGLUT2", "Hindbrain_Excitatory"),
		("@VGLUT3", "Hindbrain_Excitatory"),
		("@NBL", "Brain_Neuroblasts")
	],
	"Pons": [
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("@SER", "Brain_CholinergicMonoaminergic"),
		("@DA", "Brain_CholinergicMonoaminergic"),
		("@NOR", "Brain_CholinergicMonoaminergic"),
		("@NIPC", "Brain_Neuroblasts"),
		("@GLY", "Hindbrain_Inhibitory"),
		("@VGLUT1", "Hindbrain_Excitatory"),
		("@VGLUT2", "Hindbrain_Excitatory"),
		("@VGLUT3", "Hindbrain_Excitatory")
	],
	"Medulla": [
		("@CHOL", "Brain_CholinergicMonoaminergic"),
		("@SER", "Brain_CholinergicMonoaminergic"),
		("@DA", "Brain_CholinergicMonoaminergic"),
		("@NIPC", "Brain_Neuroblasts"),
		("@VGLUT1", "Hindbrain_Excitatory"),
		("@VGLUT2", "Hindbrain_Excitatory"),
		("@VGLUT3", "Hindbrain_Excitatory"),
		("@GLY", "Hindbrain_Inhibitory")
	],
	"SpinalCord": [
		("@NIPC", "Brain_Neuroblasts"),
		("@VGLUT1", "SpinalCord_Excitatory"),
		("@VGLUT2", "SpinalCord_Excitatory"),
		("@VGLUT3", "SpinalCord_Excitatory"),
		("@GLY", "SpinalCord_Inhibitory"),
		("@GABA", "SpinalCord_Inhibitory"),
		("PSN", "Peripheral_Neurons"),
		("*", "SpinalCord_Excitatory"),
	],
	"DRG": [
		("*", "Sensory_Neurons")
	],
	"Sympathetic": [
		("*", "Sympathetic_Neurons")
	],
	"Enteric": [
		("*", "Enteric_Neurons")
	],
}