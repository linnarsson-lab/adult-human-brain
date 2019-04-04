import os
from typing import *
import numpy as np
import loompy
import luigi
import logging
from .config import config
from .cytograph2 import Cytograph2
from .punchcard import Punchcard, PunchcardSubset
from .aggregator import Aggregator
from .workflow import workflow
from cytograph.annotation import AutoAutoAnnotator, AutoAnnotator, CellCycleAnnotator
import cytograph.plotting as cgplot
from cytograph.clustering import ClusterValidator


#
# Overview of the cytograph2 pipeline
#
# sample1 ----\                           /--> Process -> First_First.loom -------------------------------------------------|
# sample2 -----> Process -> First.loom --<                                                                                  |
# sample3 ----/                           \--> Process -> First_Second.loom ------------------------------------------------|
#                                                                                                                           |
#                                                                                                                            >---> Pool --> All.loom
#                                                                                                                           |
# sample4 ----\                            /--> Process -> Second_First.loom -----------------------------------------------|
# sample5 -----> Process -> Second.loom --<                                                                                 |
# sample6 ----/                            \                                     /--> Process -> Second_Second_First.loom --|
#                                          \--> Process -> Second_Second.loom --<                                           |
#                                                                                \--> Process -> Second_Second_Second.loom -|
#


def workflow(ds: loompy.LoomConnection, pool: str, steps: List[str], params: Dict[str, Any], output: Any) -> None:
	logging.info(f"Processing {pool}")
	cytograph = Cytograph2(**params)
	if "poisson_pooling" in steps:
		cytograph.poisson_pooling(ds)
	cytograph.fit(ds)

	if "aggregate" in steps or "export" in steps:
		logging.info(f"Aggregating {pool}")
		with output["agg"].temporary_path() as agg_file:
				Aggregator().aggregate(ds, agg_file)
				with loompy.connect(agg_file) as dsagg:
					logging.info("Computing auto-annotation")
					aa = AutoAnnotator(root=config.paths.autoannotation)
					aa.annotate_loom(dsagg)
					aa.save_in_loom(dsagg)

					logging.info("Computing auto-auto-annotation")
					n_clusters = dsagg.shape[1]
					(selected, selectivity, specificity, robustness) = AutoAutoAnnotator(n_genes=6).fit(dsagg)
					dsagg.set_attr("MarkerGenes", np.array([" ".join(ds.ra.Gene[selected[:, ix]]) for ix in np.arange(n_clusters)]), axis=1)
					# TODO: ugly	
					np.set_printoptions(precision=1, suppress=True)
					dsagg.set_attr("MarkerSelectivity", np.array([str(selectivity[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
					dsagg.set_attr("MarkerSpecificity", np.array([str(specificity[:, ix]) for ix in np.arange(n_clusters)]), axis=1)
					dsagg.set_attr("MarkerRobustness", np.array([str(robustness[:, ix]) for ix in np.arange(n_clusters)]), axis=1)

					if "export" in steps:
						logging.info(f"Reporting {pool}")
						with output["export"].temporary_path() as out_dir:
							if not os.path.exists(out_dir):
								os.mkdir(out_dir)
							logging.info("Plotting manifold graph with auto-annotation")
							tags = list(dsagg.col_attrs["AutoAnnotation"])
							cgplot.manifold(ds, os.path.join(out_dir, f"{pool}_TSNE_manifold.aa.png"), tags)
							logging.info("Plotting manifold graph with auto-auto-annotation")
							tags = list(dsagg.col_attrs["MarkerGenes"])
							cgplot.manifold(ds, os.path.join(out_dir, pool + "_TSNE_manifold.aaa.png"), tags)
							cgplot.manifold(ds, os.path.join(out_dir, pool + "_UMAP_manifold.aaa.png"), tags, embedding="UMAP")
							logging.info("Plotting marker heatmap")
							cgplot.markerheatmap(ds, dsagg, n_markers_per_cluster=10, out_file=os.path.join(out_dir, pool + "_heatmap.pdf"))
							logging.info("Plotting latent factors")
							cgplot.factors(ds, base_name=os.path.join(out_dir, pool + "_factors"))
							logging.info("Plotting cell cycle")
							CellCycleAnnotator(ds).plot_cell_cycle(os.path.join(out_dir, pool + "_cellcycle.png"))
							logging.info("Plotting markers")
							cgplot.markers(ds, out_file=os.path.join(out_dir, pool + "_markers.png"))
							logging.info("Plotting neighborhood diagnostics")
							cgplot.radius_characteristics(ds, out_file=os.path.join(out_dir, pool + "_neighborhoods.png"))
							logging.info("Plotting batch covariates")
							cgplot.batch_covariates(ds, out_file=os.path.join(out_dir, pool + "_batches.png"))
							logging.info("Plotting UMI/gene counts")
							cgplot.umi_genes(ds, out_file=os.path.join(out_dir, pool + "_umi_genes.png"))
							logging.info("Assessing cluster predictive power")
							ClusterValidator().fit(ds, os.path.join(out_dir, f"{pool}_cluster_pp.png"))
							logging.info("Plotting embedded velocity")
							cgplot.embedded_velocity(ds, out_file=os.path.join(out_dir, f"{pool}_velocity.png"))
							logging.info("Plotting TFs")
							cgplot.TFs(ds, dsagg, out_file_root=os.path.join(out_dir, pool))


class Process(luigi.Task):
	subset = luigi.Parameter()

	def output(self) -> luigi.Target:
		return {
			"loom": luigi.LocalTarget(os.path.join(config().build, "data", self.subset.longname() + ".loom")),
			"agg": luigi.LocalTarget(os.path.join(config().build, "data", self.subset.longname() + ".agg.loom")) if "aggregate" in self.subset.steps else None,
			"report": luigi.LocalTarget(os.path.join(config().build, "report", self.subset.longname())) if "report" in self.subset.steps else None
		}

	def requires(self) -> List[luigi.Task]:
		if self.card.name == "Root":
			return []
		else:
			# Get the subset in the parent Punchcard with the same name as this punchcard; these are the source cells
			source = self.card.parent.get_subset(self.card.name)
			return Process(source)

	def run(self) -> None:
		is_root = self.subset.card.name == "Root"
		with self.output()["loom"].temporary_path() as out_file:
			logging.info(f"Collecting cells for {self.subset.longname()}")
			with loompy.new(out_file) as dsout:
				if is_root:
					# TODO: collect directly from samples, optionally with doublet removal and min_umis etc.
					pass
				else:
					# Collect from a previous punchard subset
					with loompy.connect(self.input()["loom"].fn, mode="r") as ds:
						# TODO: make the right punchcard selection here
						for (ix, selection, view) in ds.scan(items=np.where(ds.ca["PC_" + self.subset.longname()] == 1)[0], axis=1, key="Accession"):
							dsout.add_columns(view.layers, view.ca, row_attrs=view.ra)
				logging.info(f"Collected {ds.shape[1]} cells")
				if self.subset.steps != []:
					steps = self.subset.steps
				elif is_root:
					steps = ["doublets", "poisson_pooling", "cells_qc", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "export"]
				else:
					steps = ["poisson_pooling", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "export"]

				workflow(dsout, self.subset.longname(), steps, {**config.params, **self.subset.params})


class Pool(luigi.Task):
	leaves = luigi.ListParameter()  # List of subsets

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(config.paths.build, "All.loom"))

	def requires(self) -> List[luigi.Task]:
		tasks = [Process(subset) for subset in self.leaves]
		return tasks

	def run(self) -> None:
		with self.output().temporary_path() as out_file:
			logging.info(f"Collecting cells for 'All.loom'")
			punchcards: List[str] = []
			clusters: List[int] = []
			next_cluster = 0
			with loompy.new(out_file) as dsout:
				for i in self.input():
					with loompy.connect(i.fn, mode="r") as ds:
						punchcards = punchcards + [os.path.basename(i.fn)[:-5]] * ds.shape[1]
						clusters = clusters + list(ds.ca.Clusters + next_cluster)
						next_cluster = max(clusters) + 1
						for (ix, selection, view) in ds.scan(axis=1, key="Accession"):
							dsout.add_columns(view.layers, view.ca, row_attrs=view.ra)
				ds.ca.Punchcard = punchcards
				ds.ca.Clusters = clusters
				workflow(dsout, "All", ["nn", "embeddings", "aggregate", "export"], {**config.params, **self.subset.params})
