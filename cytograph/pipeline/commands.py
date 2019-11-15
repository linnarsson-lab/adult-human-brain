import fnmatch
import logging
import os
import sqlite3 as sqlite
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union
from ..plotting import qc_plots
import click
import numpy as np
from loompy import create_from_fastq,connect,combine_faster
from ..preprocessing.doublet_finder import doublet_finder
from ..preprocessing import qc_functions
from .._version import __version__ as version
from .config import load_config
from .engine import CondorEngine, Engine, LocalEngine
from .punchcards import PunchcardDeck, PunchcardSubset, PunchcardView
from .workflow import (PoolWorkflow, RootWorkflow, SubsetWorkflow,
                       ViewWorkflow, Workflow)
import csv



def create_build_folders(path: str) -> None:
	Path(os.path.join(path, "data")).mkdir(exist_ok=True)
	Path(os.path.join(path, "exported")).mkdir(exist_ok=True)


@click.group()
@click.option('--build-location')
@click.option('--show-message/--hide-message', default=True)
@click.option('--verbosity', default="info", type=click.Choice(['error', 'warning', 'info', 'debug']))
def cli(build_location: str = None, show_message: bool = True, verbosity: str = "info") -> None:
	config = load_config()
	level = {"error": 40, "warning": 30, "info": 20, "debug": 10}[verbosity]
	logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
	logging.captureWarnings(True)

	# Allow command-line options to override config settings
	if build_location is not None:
		config.paths.build = build_location

	if show_message:
		print(f"Cytograph v{version} by Linnarsson Lab 🌸 (http://linnarssonlab.org)")
		if os.path.exists(config.paths.build):
			print(f"            Build: {config.paths.build}")
		else:
			print(f"            Build: {config.paths.build} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		if os.path.exists(config.paths.samples):
			print(f"          Samples: {config.paths.samples}")
		else:
			print(f"          Samples: {config.paths.samples} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		if os.path.exists(config.paths.autoannotation):
			print(f"  Auto-annotation: {config.paths.autoannotation}")
		else:
			print(f"  Auto-annotation: {config.paths.autoannotation} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		if os.path.exists(config.paths.metadata):
			print(f"         Metadata: {config.paths.metadata}")
		else:
			print(f"         Metadata: {config.paths.metadata} \033[1;31;40m-- FILE DOES NOT EXIST --\033[0m")
		print(f"   Fastq template: {config.paths.fastqs}")
		if os.path.exists(config.paths.index):
			print(f"            Index: {config.paths.index}")
		else:
			print(f"            Index: {config.paths.index} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		print()


@cli.command()
@click.option('--engine', default="local", type=click.Choice(['local', 'condor']))
@click.option('--dryrun/--no-dryrun', is_flag=True, default=False)
def build(engine: str, dryrun: bool) -> None:
	try:
		config = load_config()
		create_build_folders(config.paths.build)
		# Load the punchcard deck
		deck = PunchcardDeck(config.paths.build)

		# Create the execution engine
		execution_engine: Optional[Engine] = None
		if engine == "local":
			execution_engine = LocalEngine(deck, dryrun)
		elif engine == "condor":
			execution_engine = CondorEngine(deck, dryrun)

		# Execute the build
		assert(execution_engine is not None)
		execution_engine.execute()
	except Exception as e:
		logging.exception(f"'build' command failed: {e}")


@cli.command()
@click.argument("subset_or_view")
def process(subset_or_view: str) -> None:
	try:
		config = load_config()  # This config will not have subset-specific settings, but we need it for the build path
		create_build_folders(config.paths.build)
		logging.info(f"Processing '{subset_or_view}'")

		deck = PunchcardDeck(config.paths.build)
		subset_obj: Union[Optional[PunchcardSubset], Optional[PunchcardView]] = deck.get_subset(subset_or_view)
		if subset_obj is None:
			subset_obj = deck.get_view(subset_or_view)
			if subset_obj is None:
				logging.error(f"Subset or view {subset_or_view} not found.")
				sys.exit(1)

		if isinstance(subset_obj, PunchcardView):
			ViewWorkflow(deck, subset_obj).process()
		elif subset_obj.card.name == "Root":
			# Load the punchcard deck and process it
			RootWorkflow(deck, subset_obj).process()
		else:
			# Load the punchcard deck, find the subset, and process it
			SubsetWorkflow(deck, subset_obj).process()
	except Exception as e:
		logging.exception(f"'process' command failed: {e}")


@cli.command()
def pool() -> None:
	try:
		config = load_config()  # This config will not have subset-specific settings, but we need it for the build path
		create_build_folders(config.paths.build)
		logging.info(f"Pooling all (leaf) punchcards into 'Pool.loom'")

		# Load the punchcard deck, and pool it
		deck = PunchcardDeck(config.paths.build)
		PoolWorkflow(deck).process()
	except Exception as e:
		logging.exception(f"'pool' command failed: {e}")


@cli.command()
@click.argument("punchcard")
def subset(punchcard: str) -> None:
	config = load_config()
	deck = PunchcardDeck(config.paths.build)
	card = deck.get_card(punchcard)
	if card is None:
		logging.error(f"Punchcard {punchcard} not found.")
		sys.exit(1)

	loom_file = os.path.join(config.paths.build, "data", card.name + ".loom")
	if os.path.exists(loom_file):
		Workflow(deck, "").compute_subsets(card)
	else:
		logging.error(f"Loom file '{loom_file}' not found")


@cli.command()
@click.argument("sampleid")
@click.option('--flowcelltable', help="Tab-delimited file with SampleID, Flowcell, Lane")
def mkloom(sampleid: str, flowcelltable: str = None) -> None:
	config = load_config()
	try:
		logging.info(f"Generating loom file for '{sampleid}'")
		table = []
		if flowcelltable is not None and flowcelltable != "":
			with open(flowcelltable) as f:
				table.append(f.readline()[:-1].split("\t"))
		elif config.paths.metadata.endswith(".db"):
			db = sqlite.connect(config.paths.metadata)
			cursor = db.cursor()
			cursor.execute('''
			SELECT
			sample.name AS SampleID,
				lane.laneno AS Lane,
				flowcell.name AS Flowcell
			FROM sample, lanesample, lane, flowcell
			WHERE 
			flowcell.runstatus = "ready" AND
			sample.sampleok != "N" AND
			sample.id = lanesample.sample_id AND
			lanesample.lane_id = lane.id AND
			flowcell.id = lane.flowcell_id
			ORDER BY SampleID, Flowcell, Lane
			;''')
			records: Dict[str, Dict[str, List[str]]] = {}
			table = cursor.fetchall()
			db.close()
		else:
			logging.error("Config paths.metadata file must be an sqlite database, or 'flowcelltable' must be provided as tab-delimited file")
			sys.exit(1)

		for sid, lane, flowcell in table:
			if sid not in records:
				records[sid] = {}
			if flowcell not in records[sid]:
				records[sid][flowcell] = []
			records[sid][flowcell].append(lane)

		if sampleid not in records:
			logging.error("Sample ID not found in metadata")
			sys.exit(1)
		fastqs: List[str] = []
		for flowcell, lanes in records[sampleid].items():
			file_pattern = config.paths.fastqs.format(sampleid=sampleid, flowcell=flowcell)
			if os.path.exists(os.path.dirname(file_pattern)):
				files = os.listdir(os.path.dirname(file_pattern))
				matching_files = sorted(fnmatch.filter(files, os.path.basename(file_pattern)))
				fastqs += [os.path.join(os.path.dirname(file_pattern), f) for f in matching_files]
			else:
				logging.error(f"Directory {os.path.dirname(file_pattern)} not found; skipping some files")
		if len(fastqs) == 0:
			logging.error("No fastq files were found.")
			sys.exit(1)
		logging.info(f"Creating loom file using kallisto with {config.execution.n_cpus} threads.")
		with TemporaryDirectory(dir=os.path.join(config.paths.build, "condor")) as tempfolder:
			create_from_fastq(os.path.join(config.paths.samples, f"{sampleid}.loom"), sampleid, fastqs, config.paths.index, config.paths.metadata, config.execution.n_cpus, tempfolder, synchronous=True)
		logging.info("Done.")
	except Exception as e:
		logging.exception(f"'mkloom' command failed: {e}")

@cli.command()
@click.argument('sampleids', nargs=-1)
@click.option('--rerun', is_flag = True, help="Rerun QC on all the samples again")
@click.option('--file', help="Sampleids comma-delimited, line for replicates ")
@click.option('--fixed_th', help="Fixed TH for the Doublet Finder flag ")
def qc(sampleids: List[str] , rerun: bool = False, file: str = None, fixed_th: float = None) -> None:
	config = load_config()
	n_cells=0
	file_reader =[]
	if file is not None:
		csv_file = open(file)
		file_reader = csv.reader(csv_file, delimiter=',')
	else:
		file_reader.append(sampleids)
	
	for row in file_reader:
		sampleids = np.unique(row)
		files: List[str] = []
		good_samples: List[str] = []
		passed_qc_files: List[str] = []
		for n,sample_id in enumerate(sampleids):
			full_path = os.path.join(config.paths.samples, sample_id + ".loom")	
			if not os.path.exists(full_path):
				logging.info ('Cannot open '+sample_id+' loom file')
				continue
			logging.info(f"Examining {sample_id}.loom")
			with connect(full_path, "r+") as ds:
				if not rerun:
					if "passedQC" in ds.attrs:
						if ds.attrs.passedQC:
							files.append(full_path)
							good_samples.append(sample_id)
							passed_qc_files.append(sample_id)
							continue
						else:
							logging.warn(f"Skipping {sample_id}.loom because it didn't passed QC in previous run.")
							passed_qc_files.append(sample_id)
							continue
				#Check if the sample has enough cells above the UMI TH and add it to the doublets check 
				logging.info(f"Computing total UMIs")
				(totals, genes) = ds.map([np.sum, np.count_nonzero], axis=1)
				ds.ca["TotalUMI"] = totals
				ds.ca["NGenes"] = genes
				ds.attrs.MeanTotalUMI = np.mean(totals/ds.shape[1])
				good_cells = (totals >= config.params.min_umis)
				#Filter samples with low fraction of good cells (cells with enough UMI counts)
				if good_cells.sum() / ds.shape[1] > config.params.min_fraction_good_cells:
					files.append(full_path)
					good_samples.append(sample_id)
					ds.attrs.passedQC = True
				else:
					logging.warn(f"Skipping {sample_id}.loom because only {good_cells.sum()} of {ds.shape[1]} cells (less than {config.params.min_fraction_good_cells * 100}%) passed QC.")
					ds.attrs.passedQC = False
					continue
				# Assess demaged/dead cells using ratio of mitochondrial gene expression and unspliced reads ratio
				qc_functions.mito_genes_ratio(ds)			
				low_mito_ratio = len(np.where(ds.ca.MT_ratio<config.params.max_fraction_MT_genes)[0])/ds.shape[1]
				if(low_mito_ratio<config.params.min_fraction_good_cells):
					logging.warn(f"Possible High demaged cell ratio in {sample_id}.loom  {len(np.where(ds.ca.MT_ratio<config.params.max_fraction_MT_genes)[0])} of {ds.shape[1]} cells (less than {config.params.min_fraction_good_cells * 100}%) had low ratio of mitochondrial gene expression.")
				qc_functions.unspliced_ratio(ds, sample_name = sample_id)
				low_us_ratio = len(np.where(ds.ca.unspliced_ratio<config.params.min_fraction_unspliced_reads)[0])/ds.shape[1]
				if(low_us_ratio>1-config.params.min_fraction_good_cells):
					logging.warn(f"Possible High demaged cell ratio in {sample_id}.loom  {len(np.where(ds.ca.unspliced_ratio<config.params.min_fraction_unspliced_reads)[0])} of {ds.shape[1]} cells (more than {(1-config.params.min_fraction_good_cells) * 100}%) had low ratio of unspliced gene expression.")
				low_ngenes_ratio = len(np.where(ds.ca.NGenes/ds.ca.TotalUMI<config.params.min_fraction_genes_UMI)[0])/ds.shape[1]
				if(low_ngenes_ratio>1-config.params.min_fraction_good_cells):
					logging.warn(f"Possible High demaged cell ratio in {sample_id}.loom  {len(np.where(ds.ca.NGenes/ds.ca.TotalUMI>config.params.min_fraction_genes_UMI)[0])} of {ds.shape[1]} cells (more than {(1-config.params.min_fraction_good_cells) * 100}%) had good ratio of gene expressed vs. UMI counts.")
		#Create a combined loom file for all the good samples
		if files and len(passed_qc_files)<len(sampleids):	
			batch_name =  '-'.join(good_samples)
			out_file = os.path.join(config.paths.qc,batch_name+".loom")
			combine_faster(files, out_file, skip_attrs=["_X", "_Y", "Clusters"])
			with connect(out_file, "r+") as ds:
				logging.info("Scoring doublets using DoubletFinder")

				doublet_finder_score ,doublet_finder_flag= doublet_finder(ds, name =batch_name, use_pca=True, proportion_artificial=0.25, qc_dir = config.paths.qc, max_th = config.params.max_doubletFinder_TH, fixed_th = fixed_th )
				ds.ca["DoubletFinderScore"] = doublet_finder_score
				ds.ca["DoubletFinderFlag"] = doublet_finder_flag
			os.remove(out_file)
			nf = 0
			for n,sample_id in enumerate(good_samples):
				with connect(files[n], "r+") as ds:
					ds.ca["DoubletFinderScore"] = doublet_finder_score[nf:(nf+ds.shape[1])]
					ds.ca["DoubletFinderFlag"] = doublet_finder_flag[nf:(nf+ds.shape[1])]
					nf = nf+ds.shape[1]
					qc_plots.all_QC_plots(ds = ds, out_file =os.path.join(config.paths.qc+"/"+ sample_id+"_QC.png"))
					logging.info(f"Adding doublets attributes to sample: {sample_id}")
		elif len(passed_qc_files)==len(sampleids):
			logging.info(f"All samples in this batch already passed QC, to rerun the QC module again add --rerun to the command line")
	if file is not None:
		csv_file.close()	
	logging.info("Done.")
