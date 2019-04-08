import os
from typing import *
import numpy as np
import loompy
import luigi
import logging
import math
import pandas as pd
from .config import config
from .cytograph2 import Cytograph2
from .punchcard import Punchcard, PunchcardSubset
from .aggregator import Aggregator
from .workflow import workflow
from cytograph.annotation import AutoAutoAnnotator, AutoAnnotator, CellCycleAnnotator
import cytograph.plotting as cgplot
from cytograph.clustering import ClusterValidator
from cytograph.species import Species
from cytograph.preprocessing import Scrublet


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


def pcw(age: str) -> float:
	"""
	Parse age strings in several formats
		"8w 5d" -> 8 weeks 5 days -> 8.71 weeks
		"8.5w" -> 8 weeks 5 days -> 8.71 weeks
		"CRL44" -> 9.16 weeks

	CRL formula
	Postconception weeks = (CRL x 1.037)^0.5 x 8.052 + 23.73
	"""
	age = str(age).lower()
	age = age.strip()
	if age.startswith("crl"):
		crl = float(age[3:])
		return (math.sqrt((crl * 1.037)) * 8.052 + 21.73) / 7
	elif " " in age:
		w, d = age.split(" ")
		if not w.endswith("w"):
			raise ValueError("Invalid age string: " + age)
		if not d.endswith("d"):
			raise ValueError("Invalid age string: " + age)
		return int(w[:-1]) + int(d[:-1]) / 7
	else:
		if not age.endswith("w"):
			raise ValueError("Invalid age string: " + age)
		age = age[:-1]
		if "." in age:
			w, d = age.split(".")
		else:
			w = age
			d = "0"
		return int(w) + int(d) / 7


def get_metadata_for(sample: str) -> Dict:
	sid = "SampleID"
	metadata_file = config.paths.metadata
	if os.path.exists(metadata_file):
		# Special handling of typed column names for our database
		with open(metadata_file) as f:
			line = f.readline()
			if "SampleID:string" in line:
				sid = "SampleID:string"
		try:
			metadata = pd.read_csv(metadata_file, delimiter=";", index_col=sid, engine="python")
			attrs = metadata.loc[sample]
			if sid == "SampleID:string":
				return {k.split(":")[0]: v for k, v in metadata.loc[sample].items()}
			else:
				return {k: v for k, v in metadata.loc[sample].items()}
		except Exception as e:
			logging.info(f"Failed to load metadata because: {e}")
			raise e
	else:
		return {}


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
					AutoAnnotator(root=config.paths.autoannotation).annotate(dsagg)

					logging.info("Computing auto-auto-annotation")
					AutoAutoAnnotator(n_genes=6).annotate(dsagg)

					if "export" in steps:
						logging.info(f"Exporting {pool}")
						with output["export"].temporary_path() as out_dir:
							if not os.path.exists(out_dir):
								os.mkdir(out_dir)
							logging.info("Plotting manifold graph with auto-annotation")
							cgplot.manifold(ds, os.path.join(out_dir, f"{pool}_TSNE_manifold.aa.png"), list(dsagg.ca.AutoAnnotation))
							logging.info("Plotting manifold graph with auto-auto-annotation")
							cgplot.manifold(ds, os.path.join(out_dir, pool + "_TSNE_manifold.aaa.png"), list(dsagg.ca.MarkerGenes))
							cgplot.manifold(ds, os.path.join(out_dir, pool + "_UMAP_manifold.aaa.png"), list(dsagg.ca.MarkerGenes), embedding="UMAP")
							logging.info("Plotting marker heatmap")
							cgplot.markerheatmap(ds, dsagg, n_markers_per_cluster=10, out_file=os.path.join(out_dir, pool + "_heatmap.pdf"))
							logging.info("Plotting latent factors")
							cgplot.factors(ds, base_name=os.path.join(out_dir, pool + "_factors"))
							logging.info("Plotting cell cycle")
							cgplot.cell_cycle(ds, os.path.join(out_dir, pool + "_cellcycle.png"))
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
	subset: PunchcardSubset = luigi.Parameter()

	def output(self) -> luigi.Target:
		return {
			"loom": luigi.LocalTarget(os.path.join(config().build, "data", self.subset.longname() + ".loom")),
			"agg": luigi.LocalTarget(os.path.join(config().build, "data", self.subset.longname() + ".agg.loom")) if "aggregate" in self.subset.steps else None,
			"export": luigi.LocalTarget(os.path.join(config().build, "export", self.subset.longname())) if "export" in self.subset.steps else None
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
					# Collect directly from samples, optionally with doublet removal and min_umis etc.
					# Specification is a nested list giving batches and replicates
					# include: [[sample1, sample2], [sample3, sample4]]
					batch_id = 0
					for batch in self.subset.include:
						replicate_id = 0
						for sample_id in batch:
							full_path = os.path.join(config.paths.samples, sample_id + ".loom")
							if not os.path.exists(full_path):
								raise FileNotFoundError(f"File {full_path} not found")

							logging.info(f"Adding {sample_id}.loom")
							with loompy.connect(full_path) as ds:
								species = Species.detect(ds).name
								col_attrs = dict(ds.ca)
								metadata = get_metadata_for(sample_id)
								for key, val in metadata.items():
									logging.info("Adding metadata attribute " + key)
									col_attrs[key] = np.array([val] * ds.shape[1])
								logging.info("Adding metadata attributes SampleID, Batch, Replicate")
								col_attrs["SampleID"] = np.array([sample_id] * ds.shape[1])
								col_attrs["Batch"] = np.array([batch_id] * ds.shape[1])
								col_attrs["Replicate"] = np.array([replicate_id] * ds.shape[1])
								if "Age" in metadata and species == "Homo sapiens":
									logging.info("Adding metadata attribute PCW")
									try:
										col_attrs["PCW"] = np.array([pcw(metadata["Age"])] * ds.shape[1])
									except:
										pass
								logging.info("Marking putative doublets using Scrublet")
								data = ds[:, :].T
								doublet_scores, predicted_doublets = Scrublet(data, expected_doublet_rate=0.05).scrub_doublets()
								col_attrs["DoubletScore"] = doublet_scores
								col_attrs["DoubletFlag"] = predicted_doublets.astype("int")
								if config.params.doublets_action == "remove":
									# TODO: remove the doublets before adding to output
								logging.info(f"Appending {sample_id} ({ds.shape[1]} cells)")
								dsout.add_columns(ds.layers, col_attrs, row_attrs=ds.row_attrs)
							replicate_id += 1
						batch_id += 1
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
