import inspect
import os
from pathlib import Path
from typing import Dict, List

import yaml


class AbstractConfig:
	def merge(self, defs: Dict) -> None:
		if defs is None:
			return
		attrs = [x for x, val in inspect.getmembers(self) if not x.startswith("_") and not inspect.ismethod(val)]
		for attr in attrs:
			if attr in defs:
				setattr(self, attr, defs[attr])

	def __str__(self) -> str:
		s = []
		attrs = [(x, val) for x, val in inspect.getmembers(self) if not x.startswith("_") and not inspect.ismethod(val)]
		for (x, val) in attrs:
			s.append(f"{x}: {val}")
		return "\n".join(s)


class PathConfig(AbstractConfig):
	build: str = ""
	samples: str = ""
	autoannotation: str = ""
	metadata: str = ""


class ParamsConfig(AbstractConfig):
	k: int = 25
	k_pooling: int = 10
	n_factors: int = 96
	min_umis: int = 1500
	n_genes: int = 2000
	doublets_action: str = "remove"
	doublets_method: str = "scrublet"
	mask: List[str] = ["cellcycle", "sex", "ieg", "mt"]
	min_fraction_good_cells: float = 0.4
	skip_missing_samples: bool = False
	clusterer: str = "louvain"  # or "surprise"


class ExecutionConfig(AbstractConfig):
	n_cpus: int = 28
	n_gpus: int = 0
	memory: int = 128


class Config:
	paths = PathConfig()
	params = ParamsConfig()
	steps = ["doublets", "poisson_pooling", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "export"]
	execution = ExecutionConfig()


def merge_config(config: Config, path: str) -> None:
	if not os.path.exists(path):
		raise IOError(f"Config path {path} not found.")

	with open(path) as f:
		defs = yaml.load(f)

	if "paths" in defs:
		config.paths.merge(defs["paths"])
	if "params" in defs:
		config.params.merge(defs["params"])
	if "steps" in defs:
		config.steps = defs["steps"]
	if "execution" in defs:
		config.execution.merge(defs["execution"])


def load_config() -> Config:
	# Builtin defaults
	config = Config()
	# Home directory
	merge_config(config, os.path.join(os.path.abspath(str(Path.home())), ".cytograph"))
	# Set build folder
	if config.paths.build == "" or config.paths.build is None:
		config.paths.build = os.path.abspath(os.path.curdir)
	# Build folder
	merge_config(config, "config.yaml")
	return config
