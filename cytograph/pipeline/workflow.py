import logging
import os
import sqlite3 as sqlite
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

import cytograph.plotting as cgplot
import loompy
from cytograph.annotation import AutoAnnotator
from cytograph.clustering import ClusterValidator
from cytograph.preprocessing import doublet_finder
from cytograph.species import Species
from collections import defaultdict

from .aggregator import Aggregator
from .config import load_config
from .cytograph import Cytograph
from .punchcards import (Punchcard, PunchcardDeck, PunchcardSubset,
                         PunchcardView)
from .utils import Tempname

#
# Overview of the cytograph2 pipeline
#
# sample1 ----\                /--> First_First.loom --------------------------------------|
# sample2 -----> First.loom --<                                                            |
# sample3 ----/                \--> First_Second.loom -------------------------------------|
#                                                                                          |
#                                                                                           >---> Pool.loom
#                                                                                          |
# sample4 ----\                 /--> Second_First.loom ------------------------------------|
# sample5 -----> Second.loom --<                                                           |
# sample6 ----/                 \                          /--> Second_Second_First.loom --|
#                               \--> Second_Second.loom --<                                |
#                                                          \--> Second_Second_Second.loom -|
#


def load_sample_metadata(path: str, sample_id: str) -> Dict[str, str]:
	if not os.path.exists(path):
		raise ValueError(f"Samples metadata file '{path}' not found.")
	if path.endswith(".db"):
		# sqlite3
		with sqlite.connect(path) as db:
			cursor = db.cursor()
			cursor.execute("SELECT * FROM sample WHERE name = ?", (sample_id,))
			keys = [x[0].capitalize() for x in cursor.description]
			vals = cursor.fetchone()
			if vals is not None:
				return dict(zip(keys, vals))
			raise ValueError(f"SampleID '{sample_id}' was not found in the samples database.")
	else:
		result = {}
		with open(path) as f:
			headers = [x.lower() for x in f.readline()[:-1].split("\t")]
			if "sampleid" not in headers and 'name' not in headers:
				raise ValueError("Required column 'SampleID' or 'Name' not found in sample metadata file")
			if "sampleid" in headers:
				sample_metadata_key_idx = headers.index("sampleid")
			else:
				sample_metadata_key_idx = headers.index("name")
			sample_found = False
			for line in f:
				items = line[:-1].split("\t")
				if len(items) > sample_metadata_key_idx and items[sample_metadata_key_idx] == sample_id:
					for i, item in enumerate(items):
						result[headers[i]] = item
					sample_found = True
		if not sample_found:
			raise ValueError(f"SampleID '{sample_id}' not found in sample metadata file")
		return result


class Metadata:
	def __init__(self, path: str) -> None:
		self.sid = "SampleID"
		self.metadata = None
		if path is not None and os.path.exists(path):
			# Special handling of typed column names for our database
			with open(path) as f:
				line = f.readline()
				if "SampleID:string" in line:
					self.sid = "SampleID:string"
			try:
				self.metadata = pd.read_csv(path, delimiter=";", index_col=self.sid, engine="python")
			except Exception as e:
				logging.error(f"Failed to load metadata because: {e}")

	def get(self, sample: str) -> Dict:
		"""
		Return metadata for the given sample as a dictionary
		"""
		if self.metadata is not None:
			if self.sid == "SampleID:string":
				return {k.split(":")[0]: v for k, v in self.metadata.loc[sample].items() if pd.notnull(v)}
			else:
				return {k: v for k, v in self.metadata.loc[sample].items() if pd.notnull(v)}
		else:
			return {}


class Workflow:
	"""
	Shared workflow for every task, implementing cytograph, aggregation and plotting

	Subclasses implement workflows that vary by the way cells are collected
	"""
	def __init__(self, deck: PunchcardDeck, name: str) -> None:
		self.config = load_config()
		self.deck = deck
		self.name = name
		self.loom_file = os.path.join(self.config.paths.build, "data", name + ".loom")
		self.agg_file = os.path.join(self.config.paths.build, "data", name + ".agg.loom")
		self.export_dir = os.path.join(self.config.paths.build, "exported", name)

	def collect_cells(self, out_file: str) -> loompy.LoomConnection:
		# Override this in subclasses
		pass

	def compute_subsets(self, card: Punchcard) -> None:
		logging.info(f"Computing subset assignments for {card.name}")
		# Load auto-annotation
		annotator = AutoAnnotator(root=self.config.paths.autoannotation)
		categories_dict: Dict[str, List] = defaultdict(list)
		for d in annotator.definitions:
				for c in d.categories:
						categories_dict[c].append(d.abbreviation)
		# Load loom file
		with loompy.connect(os.path.join(self.config.paths.build, "data", card.name + ".loom"), mode="r+") as ds:
			subset_per_cell = np.zeros(ds.shape[1], dtype=object)
			taken = np.zeros(ds.shape[1], dtype=bool)
			with loompy.connect(os.path.join(self.config.paths.build, "data", card.name + ".agg.loom"), mode="r") as dsagg:
				for subset in card.subsets.values():
					logging.debug(f"{subset.name}: {subset.include}")
					selected = np.zeros(ds.shape[1], dtype=bool)
					if len(subset.include) > 0:
						# Include clusters that have any of the given tags
						for tag in subset.include:
							# If annotation is a category, check all category auto-annotations
							if tag in categories_dict.keys():
								# tag can be a list of strings only for the root punchcard, which doesn't have a loom file, so this method will never be called on it, hence we can ignore the type error
								for aa in categories_dict[tag]:  # type: ignore
									for ix in range(dsagg.shape[1]):
										if aa in dsagg.ca.AutoAnnotation[ix].split(" "):
											selected = selected | (ds.ca.Clusters == ix)
							else:
								for ix in range(dsagg.shape[1]):
									if tag in dsagg.ca.AutoAnnotation[ix].split(" "):
										selected = selected | (ds.ca.Clusters == ix)
					else:
						selected = ~taken
					# Exclude cells that don't match the onlyif expression
					if subset.onlyif != "" and subset.onlyif is not None:
						selected = selected & eval(subset.onlyif, globals(), {k: v for k, v in ds.ca.items()})
					# Don't include cells that were already taken
					selected = selected & ~taken
					subset_per_cell[selected] = subset.name
					taken[selected] = True
					logging.debug(f"Selected {selected.sum()} cells")
				ds.ca.Subset = subset_per_cell
				# plot subsets
				parent_dir = os.path.join(self.config.paths.build, "exported", card.name)
				if os.path.exists(parent_dir):
					cgplot.punchcard_selection(ds, os.path.join(parent_dir, f"{card.name}_subsets.png"), list(dsagg.ca.MarkerGenes), list(dsagg.ca.AutoAnnotation))

	def process(self) -> None:
		config = load_config()
		# STEP 1: build the .loom file and perform manifold learning (Cytograph)
		# Maybe we're already done?
		if os.path.exists(self.loom_file):
			logging.info(f"Skipping '{self.name}.loom' because it was already complete.")
		elif os.path.exists(self.loom_file + '.rerun'):
			with loompy.connect(self.loom_file + '.rerun') as ds:
				logging.info(f"Repeating cytograph on {ds.shape[1]} previously collected cells")
				ds.attrs.config = config.to_string()
				Cytograph(config=self.config).fit(ds)
				os.rename(self.loom_file + '.rerun', self.loom_file)
		else:
			with Tempname(self.loom_file) as out_file:	
				self.collect_cells(out_file)
				with loompy.connect(out_file) as ds:
					ds.attrs.config = config.to_string()
					logging.info(f"Collected {ds.shape[1]} cells")
					Cytograph(config=self.config).fit(ds)

		# STEP 2: aggregate and create the .agg.loom file
		if os.path.exists(self.agg_file):
			logging.info(f"Skipping '{self.name}.agg.loom' because it was already complete.")
		else:
			with loompy.connect(self.loom_file) as dsout:
				clusts, labels = np.unique(dsout.ca.Clusters, return_inverse=True)
				if len(np.unique(clusts)) != dsout.ca.Clusters.max() + 1:
					logging.info(f"Renumbering clusters before aggregating.")
					dsout.ca.ClustersCollected = dsout.ca.Clusters
					dsout.ca.Clusters = labels
				with Tempname(self.agg_file) as out_file:
					Aggregator(mask=Species.detect(dsout).mask(dsout, config.params.mask)).aggregate(dsout, out_file=out_file)
				with loompy.connect(self.agg_file) as dsagg:
					dsagg.attrs.config = config.to_string()

		# STEP 3: export plots
		if os.path.exists(self.export_dir):
			logging.info(f"Skipping 'exported/{self.name}' because it was already complete.")
		else:
			pool = self.name
			logging.info(f"Exporting plots for {pool}")
			with Tempname(self.export_dir) as out_dir:
				os.mkdir(out_dir)
				with loompy.connect(self.loom_file) as ds:
					with loompy.connect(self.agg_file) as dsagg:
						if dsagg.shape[1] > 1:
							cgplot.manifold(ds, os.path.join(out_dir, f"{pool}_TSNE_manifold.png"), list(dsagg.ca.MarkerGenes), list(dsagg.ca.AutoAnnotation))
							if "UMAP" in ds.ca:
								cgplot.manifold(ds, os.path.join(out_dir, pool + "_UMAP_manifold.png"), list(dsagg.ca.MarkerGenes), list(dsagg.ca.AutoAnnotation), embedding="UMAP")
							if "PCA" in ds.ca:
								cgplot.manifold(ds, os.path.join(out_dir, pool + "_PCA_manifold.png"), list(dsagg.ca.MarkerGenes), list(dsagg.ca.AutoAnnotation), embedding="PCA")
							if ds.ca.Clusters.max() < 500:
								cgplot.markerheatmap(ds, dsagg, out_file=os.path.join(out_dir, pool + "_markers_pooled_heatmap.pdf"), layer="pooled")
								cgplot.markerheatmap(ds, dsagg, out_file=os.path.join(out_dir, pool + "_markers_heatmap.pdf"), layer="")
							if "HPF" in ds.ca:
								cgplot.factors(ds, base_name=os.path.join(out_dir, pool + "_factors"))
							if "CellCycle_G1" in ds.ca:
								cgplot.cell_cycle(ds, os.path.join(out_dir, pool + "_cellcycle.png"))
							if "KNN" in ds.col_graphs:
								cgplot.radius_characteristics(ds, out_file=os.path.join(out_dir, pool + "_neighborhoods.png"))
							cgplot.batch_covariates(ds, out_file=os.path.join(out_dir, pool + "_batches.png"))
							cgplot.umi_genes(ds, out_file=os.path.join(out_dir, pool + "_umi_genes.png"))
							if ds.ca.Clusters.max() < 500:
								cgplot.TF_heatmap(ds, dsagg, out_file=os.path.join(out_dir, f"{pool}_TFs_pooled_heatmap.pdf"), layer="pooled")
								cgplot.TF_heatmap(ds, dsagg, out_file=os.path.join(out_dir, f"{pool}_TFs_heatmap.pdf"), layer="")
							if "GA" in dsagg.col_graphs:
								cgplot.metromap(ds, dsagg, out_file=os.path.join(out_dir, f"{pool}_metromap.png"))
							if "cluster-validation" in self.config.steps:
								ClusterValidator().fit(ds, os.path.join(out_dir, f"{pool}_cluster_pp.png"))
							if "unspliced_ratio" in ds.ca:
								cgplot.attrs_on_TSNE(
									ds = ds,
									out_file=os.path.join(out_dir, f"{pool}_QC.png"), 
									attrs=["DoubletFinderFlag", "DoubletFinderScore", "TotalUMI", "NGenes", "unspliced_ratio", "MT_ratio"], 
									plot_title=["Doublet Flag", "Doublet Score", "UMI counts", "Number of genes", "Unspliced / Total UMI", "Mitochondrial / Total UMI"])

		# If there's a punchcard for this subset, go ahead and compute the subsets for that card
		card_for_subset = self.deck.get_card(self.name)
		if card_for_subset is not None:
			self.compute_subsets(card_for_subset)

		logging.info("Done.")


class RootWorkflow(Workflow):
	"""
	A workflow for the root, which collects its cells directly from input samples
	"""
	def __init__(self, deck: PunchcardDeck, subset: PunchcardSubset) -> None:
		super().__init__(deck, subset.longname())
		self.subset = subset
		self.deck = deck
		self.config = load_config(subset)

	def collect_cells(self, out_file: str) -> None:
		# Make sure all the sample files exist
		err = False
		missing_samples: List[str] = []
		if type(self.subset.include[0]) is list:
			include = [item for sublist in self.subset.include for item in sublist]  # Flatten the list (see https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists)
		else:
			include = self.subset.include  # type: ignore
		for sample_id in include:
			full_path = os.path.join(self.config.paths.samples, sample_id + ".loom")
			if not os.path.exists(full_path):
				logging.error(f"Sample file '{full_path}' not found")
				err = True
				missing_samples.append(sample_id)
		if err and not self.config.params.skip_missing_samples:
			sys.exit(1)

		# metadata = Metadata(self.config.paths.metadata)
		logging.info(f"Collecting cells for {self.name}")
		logging.debug(out_file)
		n_cells = 0
		files: List[str] = []
		selections: List[str] = []
		new_col_attrs: List[Dict[str, np.ndarray]] = []

		for sample_id in include:
			full_path = os.path.join(self.config.paths.samples, sample_id + ".loom")
			if not os.path.exists(full_path):
				continue
			logging.info(f"Examining {sample_id}.loom")
			if not self.config.params.skip_metadata:
				metadata = load_sample_metadata(self.config.paths.metadata, sample_id)
			with loompy.connect(full_path, "r") as ds:
				if self.config.params.passedQC and not ds.attrs.passedQC :
					logging.warn(f"Skipping {sample_id}.loom - did not pass QC.")
					continue
				species = Species.detect(ds).name
				col_attrs = dict(ds.ca)
				if not self.config.params.skip_metadata:
					for key, val in metadata.items():
						col_attrs[key] = np.array([val] * ds.shape[1])
				if "Species" not in col_attrs:
					col_attrs["Species"] = np.array([species] * ds.shape[1])
				col_attrs["SampleID"] = np.array([sample_id] * ds.shape[1])
				for attr, val in ds.attrs.items():
					if attr == "LOOM_SPEC_VERSION" or attr in col_attrs:
						continue
					col_attrs[attr] = np.array([val] * ds.shape[1])
				if "doublets" in self.config.steps or self.config.params.passedQC :
					if "DoubletFinderFlag" not in ds.ca:
						logging.info("Scoring doublets using DoubletFinder")
						doublet_finder_score, predicted_doublets = doublet_finder(ds,graphs = False)
						col_attrs["DoubletFinderScore"] = doublet_finder_score 
						col_attrs["DoubletFinderFlag"] = predicted_doublets
					else:
						logging.info("Using QC DoubletFinder flag")
						predicted_doublets  = ds.ca.DoubletFinderFlag
				if "TotalUMI" not in ds.ca or "NGenes" not in ds.ca:		
					logging.info(f"Computing total UMIs")
					(totals, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
					col_attrs["TotalUMI"] = totals
					col_attrs["NGenes"] = genes
					good_cells = (totals >= self.config.params.min_umis)
				else:
					good_cells = (ds.ca.TotalUMI >= self.config.params.min_umis)
				
				if ("doublets" in self.config.steps or self.config.params.passedQC) and self.config.params.doublets_action == "remove":
					logging.info(f"Removing {np.sum(predicted_doublets>0)} doublets and {(~good_cells).sum()} cells with <{self.config.params.min_umis} UMIs")
					good_cells = np.logical_and(good_cells , predicted_doublets==0)
				else:
					logging.info(f"Removing {(~good_cells).sum()} cells with <{self.config.params.min_umis} UMIs")					
				
				if good_cells.sum() / ds.shape[1] > self.config.params.min_fraction_good_cells:
					logging.info(f"Including {good_cells.sum()} of {ds.shape[1]} cells")
					n_cells += good_cells.sum()
					files.append(full_path)
					selections.append(good_cells)
					new_col_attrs.append(col_attrs)
				else:
					logging.warn(f"Skipping {sample_id}.loom because only {good_cells.sum()} of {ds.shape[1]} cells (less than {self.config.params.min_fraction_good_cells * 100}%) passed QC.")
		logging.info(f"Collecting a total of {n_cells} cells.")
		loompy.combine_faster(files, out_file, {}, selections, key="Accession", skip_attrs=["_X", "_Y", "Clusters"])
		logging.info(f"Adding column attributes")
		with loompy.connect(out_file) as ds:
			for attr in new_col_attrs[0].keys():
				if all([attr in attrs for attrs in new_col_attrs]):
					ds.ca[attr] = np.concatenate([x[attr][sel] for x, sel in zip(new_col_attrs, selections)])
				else:
					logging.warn(f"Skipping column attribute {attr} because it was only present in some of the inputs")
			ds.attrs['Species'] = species	
	
class SubsetWorkflow(Workflow):
	"""
	A workflow for intermediate punchcards, which collects its cells from a previous punchcard subset
	"""
	def __init__(self, deck: PunchcardDeck, subset: PunchcardSubset) -> None:
		super().__init__(deck, subset.longname())
		self.subset = subset
		self.deck = deck
		self.config = load_config(subset)

	def collect_cells(self, out_file: str) -> None:
		# Verify that the previous punchard subset exists
		parent = os.path.join(self.config.paths.build, "data", self.subset.card.name + ".loom")
		if not os.path.exists(parent):
			logging.error(f"Punchcard file '{parent}' was missing.")
			sys.exit(1)

		# Verify that there are some cells in the subset
		with loompy.connect(parent, mode="r") as ds:
			cells = (ds.ca.Subset == self.subset.name)
			if cells.sum() == 0:
				logging.info(f"Skipping {self.name} because the subset was empty")
				sys.exit(0)

		logging.info(f"Collecting cells for {self.name}")
		# Collect from a previous punchard subset
		loompy.combine_faster([parent], out_file, None, selections=[cells], key="Accession")


class ViewWorkflow(Workflow):
	"""
	A workflow for views, which collects its cells from arbitrary punchcard subsets

	Views
			Collect cells from punchcard subsets and/or from the pool
			Cannot collect from other views
			Can use include and onlyif to filter the sources
			Adds column attributes Source (the name of the source punchcard) and SourceClusters
	"""
	def __init__(self, deck: PunchcardDeck, view: PunchcardView) -> None:
		super().__init__(deck, view.name)
		self.view = view
		self.deck = deck
		self.config = load_config(view)

	def _compute_cells_for_view(self, subset: str, include: List[str], onlyif: str) -> np.ndarray:
		# Load auto-annotation
		annotator = AutoAnnotator(root=self.config.paths.autoannotation)
		categories_dict: Dict[str, List] = defaultdict(list)
		for d in annotator.definitions:
				for c in d.categories:
						categories_dict[c].append(d.abbreviation)
		# Load loom file
		with loompy.connect(os.path.join(self.config.paths.build, "data", subset + ".loom"), mode="r") as ds:
			with loompy.connect(os.path.join(self.config.paths.build, "data", subset + ".agg.loom"), mode="r") as dsagg:
				selected = np.zeros(ds.shape[1], dtype=bool)
				if len(include) > 0:
					# Include clusters that have any of the given tags
					for tag in include:
						# If annotation is a category, check all category auto-annotations
						if tag in categories_dict.keys():
							# tag can be a list of strings only for the root punchcard, which doesn't have a loom file, so this method will never be called on it, hence we can ignore the type error
							for aa in categories_dict[tag]:  # type: ignore
								for ix in range(dsagg.shape[1]):
									if aa in dsagg.ca.AutoAnnotation[ix].split(" "):
										selected = selected | (ds.ca.Clusters == ix)
						else:
							for ix in range(dsagg.shape[1]):
								if tag in dsagg.ca.AutoAnnotation[ix].split(" "):
									selected = selected | (ds.ca.Clusters == ix)
				else:
					selected = np.ones(ds.shape[1], dtype=bool)
				# Exclude cells that don't match the onlyif expression
				if onlyif != "" and onlyif is not None:
					selected = selected & eval(onlyif, globals(), {k: v for k, v in ds.ca.items()})
				return selected

	def collect_cells(self, out_file: str) -> None:
		# Verify that the previous punchard subsets exist
		for s in self.view.sources:
			parent = os.path.join(self.config.paths.build, "data", s + ".loom")
			if not os.path.exists(parent):
				logging.error(f"Punchcard file '{parent}' was missing.")
				sys.exit(1)
		
		logging.info(f"Collecting cells for {self.name}")
		files = []
		selections = []
		previous_clusters = []
		previous_file = []
		for s in self.view.sources:
			logging.info(f"Collecting cells from '{s}'")
			f = os.path.join(self.config.paths.build, "data", s + ".loom")
			selected = self._compute_cells_for_view(s, self.view.include, self.view.onlyif)
			logging.info(f"Collecting {selected.sum()} cells from '{s}'")
			files.append(f)
			selections.append(selected)
			with loompy.connect(f, mode="r") as ds:
				previous_clusters.append(ds.ca.Clusters[selected])
			previous_file.append([s] * selected.sum())
		logging.debug("Combining files")
		loompy.combine_faster(files, out_file, None, selections, key="Accession")
		with loompy.connect(out_file) as ds:
			ds.ca.SourceClusters = np.concatenate(previous_clusters)
			ds.ca.Source = np.concatenate(previous_file)


class PoolWorkflow(Workflow):
	"""
	Workflow for the final pooling step, which collects its cells from all the leaf subsets
	"""
	def __init__(self, deck: PunchcardDeck) -> None:
		super().__init__(deck, "Pool")
		self.deck = deck
		self.config = load_config()
		# Merge pool-specific config
		if os.path.exists(os.path.join(self.config.paths.build, "pool_config.yaml")):
			self.config.merge_with(os.path.join(self.config.paths.build, "pool_config.yaml"))

	def collect_cells(self, out_file: str) -> None:
		punchcards: List[str] = []
		clusters: List[int] = []
		punchcard_clusters: List[int] = []
		next_cluster = 0

		# Check that all the inputs exist
		logging.info(f"Checking that all input files are present")
		err = False
		for subset in self.deck.get_leaves():
			if not os.path.exists(os.path.join(self.config.paths.build, "data", subset.longname() + ".loom")):
				logging.error(f"Punchcard file 'data/{subset.longname()}.loom' is missing")
				err = True
		if err:
			sys.exit(1)

		type_dict = defaultdict(list)
		shape_dict = defaultdict(list)
		for subset in self.deck.get_leaves():
			logging.info(f"Collecting metadata from {subset.longname()}")
			with loompy.connect(os.path.join(self.config.paths.build, "data", subset.longname() + ".loom"), mode="r") as ds:
				punchcards = punchcards + [subset.longname()] * ds.shape[1]
				punchcard_clusters = punchcard_clusters + list(ds.ca.Clusters)
				clusters = clusters + list(ds.ca.Clusters + next_cluster)
				next_cluster = max(clusters) + 1
				for k, v in ds.ca.items():
					type_dict[k].append(v.dtype)
					sh = 0 if len(v.shape) == 1 else v.shape[1]
					shape_dict[k].append(sh)
		skip_attrs = [k for k, v in type_dict.items() if not len(set(v)) == 1 or not len(set(shape_dict[k])) == 1]
		logging.info(f"Skipping attrs: {skip_attrs}")

		logging.info(f"Collecting all cells into {out_file}")
		files = [os.path.join(self.config.paths.build, "data", subset.longname() + ".loom") for subset in self.deck.get_leaves()]
		loompy.combine_faster(files, out_file, None, key="Accession", skip_attrs=skip_attrs)

		with loompy.connect(out_file) as dsout:
			dsout.ca.Punchcard = punchcards
			dsout.ca.PunchcardClusters = punchcard_clusters
			dsout.ca.Clusters = clusters
			logging.info(f"{dsout.ca.Clusters.max() + 1} clusters")
