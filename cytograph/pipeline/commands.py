import fnmatch
import logging
import os
import sqlite3 as sqlite
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import click

from loompy import create_from_fastq

from .._version import __version__ as version
from .config import load_config
from .engine import CondorEngine, Engine, LocalEngine
from .punchcards import PunchcardDeck, PunchcardSubset, PunchcardView
from .workflow import PoolWorkflow, RootWorkflow, SubsetWorkflow, ViewWorkflow, Workflow


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

	create_build_folders(config.paths.build)

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
@clock.option('--tempfolder')
def mkloom(sampleid: str, flowcelltable: str = None, tempfolder: str = None) -> None:
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
			for lane in lanes:
				file_pattern = config.paths.fastqs.format(sampleid=sampleid, flowcell=flowcell, lane=lane)
				if os.path.exists(os.path.dirname(file_pattern)):
					files = os.listdir(os.path.dirname(file_pattern))
					matching_files = sorted(fnmatch.filter(files, os.path.basename(file_pattern)))
					if len(matching_files) != 2:
						logging.error("Config paths.fastqs must match exactly two files per sampleID, flowcell and lane")
					fastqs += [os.path.join(os.path.dirname(file_pattern), f) for f in matching_files]
				else:
					logging.error(f"Directory {os.path.dirname(file_pattern)} not found; skipping some files")
		if len(fastqs) == 0:
			logging.error("No fastq files were found.")
			sys.exit(1)
		logging.info(f"Creating loom file using kallisto with {config.execution.n_cpus} threads.")
		create_from_fastq(os.path.join(config.paths.samples, f"{sampleid}.loom"), sampleid, fastqs, config.paths.index, config.paths.metadata, config.execution.n_cpus, tempfolder)
		logging.info("Done.")
	except Exception as e:
		logging.exception(f"'mkloom' command failed: {e}")
