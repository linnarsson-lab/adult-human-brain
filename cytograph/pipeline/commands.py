import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

import click

from .._version import __version__ as version
from .config import config, merge_config
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
	level = {"error": 40, "warning": 30, "info": 20, "debug": 10}[verbosity]
	logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
	logging.captureWarnings(True)

	# Allow command-line options to override config settings
	if build_location is not None:
		config.paths.build = build_location

	create_build_folders(config.paths.build)

	if show_message:
		print(f"Cytograph v{version} by Linnarsson Lab (http://linnarssonlab.org)")
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
			print(f"         Metadata: {config.paths.metadata} \033[1;31;40m-- DIRECTORY DOES NOT EXIST --\033[0m")
		print(f"            Steps: {', '.join(config.steps)}")
		print()


@cli.command()
@click.option('--engine', default="local", type=click.Choice(['local', 'condor']))
@click.option('--dryrun/--no-dryrun', is_flag=True, default=False)
def build(engine: str, dryrun: bool) -> None:
	try:
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
	except Exception:
		logging.exception("'build' command failed")


@cli.command()
@click.argument("subset_or_view")
def process(subset_or_view: str) -> None:
	try:
		logging.info(f"Processing '{subset_or_view}'")

		deck = PunchcardDeck(config.paths.build)
		subset_obj: Union[Optional[PunchcardSubset], Optional[PunchcardView]] = deck.get_subset(subset_or_view)
		if subset_obj is None:
			subset_obj = deck.get_view(subset_or_view)
			if subset_obj is None:
				logging.error(f"Subset or view {subset_or_view} not found.")
				sys.exit(1)

		# Merge any subset-specific configs
		config.params.merge(subset_obj.params)
		if subset_obj.steps != [] and subset_obj.steps is not None:
			config.steps = subset_obj.steps
		config.execution.merge(subset_obj.execution)

		if isinstance(subset_obj, PunchcardView):
			ViewWorkflow(deck, subset_obj).process()
		elif subset_obj.card.name == "Root":
			# Load the punchcard deck and process it
			RootWorkflow(deck, subset_obj).process()
		else:
			# Load the punchcard deck, find the subset, and process it
			SubsetWorkflow(deck, subset_obj).process()
	except Exception:
		logging.exception("'process' command failed")


@cli.command()
def pool() -> None:
	try:
		logging.info(f"Pooling all (leaf) punchcards into 'Pool.loom'")

		# Merge pool-specific config
		merge_config(config, os.path.join(config.paths.build, "pool_config.yaml"))

		# Load the punchcard deck, and pool it
		deck = PunchcardDeck(config.paths.build)
		PoolWorkflow(deck).process()
	except Exception:
		logging.exception("'pool' command failed")


@cli.command()
@click.argument("punchcard")
def subset(punchcard: str) -> None:
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
