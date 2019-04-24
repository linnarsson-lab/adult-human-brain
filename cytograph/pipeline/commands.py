import sys
import os
import click
import logging
from pathlib import Path
from .config import config, merge_config
from .punchcards import PunchcardDeck
from .engine import LocalEngine, CondorEngine, Engine
from .workflow import pool_leaves, process_subset, process_root
from .._version import __version__ as version


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
@click.option('--execution-engine')
@click.option('--dryrun/--no-dryrun', is_flag=True, default=config.execution.dryrun)
def build(execution_engine: str = None, dryrun: bool = None) -> None:
	# Allow command-line options to override config settings
	if execution_engine is not None:
		config.execution.engine = execution_engine
	if dryrun is not None:
		config.execution.dryrun = dryrun

	# Load the punchcard deck
	deck = PunchcardDeck(config.paths.build)

	# Create the execution engine
	engine: Engine = None
	if config.execution.engine == "local":
		engine = LocalEngine(deck)
	elif config.execution.engine == "condor":
		engine = CondorEngine(deck)
	else:
		raise ValueError(f"Invalid execution engine '{config.execution.engine}'")

	# Execute the build
	engine.execute()


@cli.command()
@click.argument("subset")
def process(subset: str) -> None:
	logging.info(f"Processing '{subset}'")

	deck = PunchcardDeck(config.paths.build)
	subset_obj = deck.get_subset(subset)
	if subset_obj is None:
		logging.error(f"Subset {subset} not found.")
		sys.exit(1)

	# Merge any subset-specific configs
	config.params.merge(subset_obj.params)
	if subset_obj.steps != [] and subset_obj.steps is not None:
		config.steps = subsubset_objset.steps
	config.execution.merge(subset_obj.execution)

	if subset_obj.card.name == "Root":
		# Load the punchcard deck and process it
		process_root(deck, subset_obj)
	else:
		# Load the punchcard deck, find the subset, and process it
		process_subset(deck, subset_obj)


@cli.command()
def pool() -> None:
	logging.info(f"Pooling all (leaf) punchcards into 'Pool.loom'")

	# Merge pool-specific config
	merge_config(config, os.path.join(config.paths.build, "Pool.yaml"))

	# Load the punchcard deck, and pool it
	deck = PunchcardDeck(config.paths.build)
	pool_leaves(deck)
