import os
from typing import *
import numpy as np
import loompy
import luigi
from pathlib import Path
import logging
import math
import random
import sys
import pandas as pd
import click
import cytograph.plotting as cgplot
from cytograph.annotation import AutoAutoAnnotator, AutoAnnotator, CellCycleAnnotator
from cytograph.clustering import ClusterValidator
from cytograph.species import Species
from cytograph.preprocessing import Scrublet
from .config import config
from .cytograph import Cytograph
from .punchcards import Punchcard, PunchcardSubset, PunchcardDeck
from .aggregator import Aggregator
from .utils import Tempname

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


# TODO: turn this into a proper class and make the API nicer
def get_metadata_for(sample: str) -> Dict:
	if config.paths.metadata is None or not os.path.exists(config.paths.metadata):
		return {}
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


def compute_subsets(card: Punchcard) -> None:
	with loompy.connect(os.path.join(config.paths.build, "data", card.name + ".loom"), mode="r+") as ds:
		subset_per_cell = np.zeros(ds.shape[1], dtype=object)
		taken = np.zeros(ds.shape[1], dtype=bool)
		with loompy.connect(os.path.join(config.paths.build, "data", card.name + ".agg.loom"), mode="r") as dsagg:
			for subset in card.subsets:
				selected = np.zeros(ds.shape[1], dtype=bool)
				if len(subset.include > 0):
					# Include clusters that have any of the given auto-annotations
					for aa in subset.include:
						for ix in range(dsagg.shape[1]):
							if aa in dsagg.ca.AutoAnnotation.split(" "):
								selected = selected | (ds.ca.Clusters == ix)
					# Exclude cells that don't match the onlyif expression
					if subset.onlyif != "":
						selected = selected & eval(subset.onlyif, globals(), ds.ca)
				else:
					selected = ~taken
				# Don't include cells that were already taken
				selected = selected & ~taken
				subset_per_cell[selected] = subset.name
		ds.ca.Subset = subset_per_cell


def process_root(deck: PunchcardDeck, subset: PunchcardSubset) -> None:
	# Collect directly from samples, optionally with doublet removal and min_umis etc.
	# Specification is a nested list giving batches and replicates
	# include: [[sample1, sample2], [sample3, sample4]]

	# Make sure all the sample files exist
	err = False
	for batch in subset.include:
		for sample_id in batch:
			full_path = os.path.join(config.paths.samples, sample_id + ".loom")
			if not os.path.exists(full_path):
				logging.error(f"Sample file '{full_path}' not found")
				err = True
	if err:
		sys.exit(1)

	if config.params.doublets_method != "scrublet":
		logging.error("Only doublets_method == 'scrublet' is allowed.")
		sys.exit(1)

	with Tempname(os.path.join(config.paths.build, "data", subset.longname() + ".loom")) as out_file:
		logging.info(f"Collecting cells for {subset.longname()}")
		logging.debug(out_file)
		with loompy.new(out_file) as dsout:
			batch_id = 0
			for batch in subset.include:
				replicate_id = 0
				for sample_id in batch:
					full_path = os.path.join(config.paths.samples, sample_id + ".loom")
					logging.info(f"Adding {sample_id}.loom")
					with loompy.connect(full_path) as ds:
						species = Species.detect(ds).name
						col_attrs = dict(ds.ca)
						metadata = get_metadata_for(sample_id)
						for key, val in metadata.items():
							col_attrs[key] = np.array([val] * ds.shape[1])
						col_attrs["SampleID"] = np.array([sample_id] * ds.shape[1])
						col_attrs["Batch"] = np.array([batch_id] * ds.shape[1])
						col_attrs["Replicate"] = np.array([replicate_id] * ds.shape[1])
						if "Age" in metadata and species == "Homo sapiens":
							try:
								col_attrs["PCW"] = np.array([pcw(metadata["Age"])] * ds.shape[1])
							except:
								pass
						if config.params.doublets_action == "remove":
							logging.info(f"Removing putative doublets using '{config.params.doublets_method}'")
						else:
							logging.info(f"Scoring putative doublets using '{config.params.doublets_method}'")
						data = ds[:, :].T
						doublet_scores, predicted_doublets = Scrublet(data, expected_doublet_rate=0.05).scrub_doublets()
						col_attrs["DoubletScore"] = doublet_scores
						col_attrs["DoubletFlag"] = predicted_doublets.astype("int")
						logging.info(f"Computing total UMIs")
						totals = ds.map([np.sum], axis=1)[0]
						ds.ca.TotalUMI = totals
						good_cells = (totals >= config.params.min_umis)
						if config.params.doublets_action == "remove":
							logging.info(f"Removing {predicted_doublets.sum()} doublets and {(~good_cells).sum()} cells with <{config.params.min_umis} UMIs")
							good_cells = good_cells & (~predicted_doublets)
						logging.info(f"Collecting {good_cells.sum()} of {data.shape[0]} cells")
						dsout.add_columns(ds.layers[:, good_cells], {att: vals[good_cells] for att, vals in col_attrs.items()}, row_attrs=ds.row_attrs)
					replicate_id += 1
				batch_id += 1
			Cytograph(steps=config.steps).fit(dsout)
			Aggregator(mask=Species.detect(dsout).mask(dsout, config.params.mask)).aggregate(dsout, agg_file=os.path.join(config.paths.build, "data", subset.longname() + ".agg.loom"), export_dir=os.path.join(config.paths.build, "exported"))
	compute_subsets(deck.get_card("Root"))


def process_subset(deck: PunchcardDeck, subset: PunchcardSubset) -> None:
	# Verify that the previous punchard subset exists
	parent = os.path.join(config.paths.build, "data", subset.card.name + ".loom")
	if not os.path.exists(parent):
		logging.error(f"Punchcard file '{parent}' was missing.")
		sys.exit(1)

	with Tempname(os.path.join(config.paths.build, "data", subset.longname() + ".loom")) as out_file:
		logging.info(f"Collecting cells for {subset.longname()}")
		with loompy.new(out_file) as dsout:
			# Collect from a previous punchard subset
			with loompy.connect(parent, mode="r") as ds:
				for (ix, selection, view) in ds.scan(items=(ds.ca.Subset == subset.longname()), axis=1, key="Accession"):
					dsout.add_columns(view.layers, view.ca, row_attrs=view.ra)

			logging.info(f"Collected {ds.shape[1]} cells")
			if subset.steps != []:
				steps = subset.steps
			elif is_root:
				steps = ["doublets", "poisson_pooling", "cells_qc", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "export"]
			else:
				steps = ["poisson_pooling", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "export"]

			Cytograph(steps=config.steps).fit(dsout)
			Aggregator(mask=Species.detect(dsout).mask(dsout, config.params.mask)).aggregate(dsout, agg_file=os.path.join(config.paths.build, "data", subset.longname() + ".agg.loom"), export_dir=os.path.join(config.paths.build, "exported"))
	compute_subsets(deck.get_card(subset.longname()))


def pool_leaves(deck: PunchcardDeck) -> None:
	with Tempname(os.path.join(config.paths.build, "data", "Pool.loom")) as out_file:
		logging.info(f"Collecting cells for 'Pool.loom'")
		punchcards: List[str] = []
		clusters: List[int] = []
		punchcard_clusters: List[int] = []
		next_cluster = 0

		# Check that all the inputs exist
		err = False
		for subset in deck.get_leaves():
			if not os.path.exists(os.path.join(config.paths.build, "data", subset.longname() + ".loom")):
				logging.error(f"Punchcard file 'data/{subset.longname()}.loom' is missing")
				err = True
		if err:
			sys.exit(1)

		with loompy.new(out_file) as dsout:
			for subset in deck.get_leaves():
				with loompy.connect(os.path.join(config.paths.build, "data", subset.longname() + ".loom"), mode="r") as ds:
					punchcards = punchcards + [subset.longname()] * ds.shape[1]
					punchcard_clusters = punchcard_clusters + list(ds.ca.Clusters)
					clusters = clusters + list(ds.ca.Clusters + next_cluster)
					next_cluster = max(clusters) + 1
					for (ix, selection, view) in ds.scan(axis=1, key="Accession"):
						dsout.add_columns(view.layers, view.ca, row_attrs=view.ra)
			ds.ca.Punchcard = punchcards
			ds.ca.PunchcardClusters = punchcard_clusters
			ds.ca.Clusters = clusters
			Cytograph(steps=["nn", "embeddings", "aggregate", "export"]).fit(dsout)
			Aggregator(mask=Species.detect(dsout).mask(dsout, config.params.mask)).aggregate(dsout, agg_file=os.path.join(config.paths.build, "data", "Pool.agg.loom"), export_dir=os.path.join(config.paths.build, "All_exported"))


def create_build_folders(path) -> None:
	Path(os.path.join(path, "data")).mkdir(exist_ok=True)
	Path(os.path.join(path, "exported")).mkdir(exist_ok=True)
