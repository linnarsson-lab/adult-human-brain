import inspect
import os
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
from types import SimpleNamespace

import yaml


def merge_namespaces(a: SimpleNamespace, b: SimpleNamespace) -> None:
	for k, v in vars(b).items():
		if isinstance(v, SimpleNamespace):
			merge_namespaces(a.__dict__[k], v)
		else:
			a.__dict__[k] = v


def merge_config(config: SimpleNamespace, path: str) -> None:
	if not os.path.exists(path):
		raise IOError(f"Config path {path} not found.")

	with open(path) as f:
		defs = yaml.load(f)

	if "paths" in defs:
		merge_namespaces(config.paths, SimpleNamespace(**defs["paths"]))
	if "params" in defs:
		merge_namespaces(config.params, SimpleNamespace(**defs["params"]))
	if "steps" in defs:
		config.steps = defs["steps"]
	if "execution" in defs:
		merge_namespaces(config.execution, SimpleNamespace(**defs["execution"]))


class Config(SimpleNamespace):
	def to_string(self, offset: int = 0) -> str:
		s = ""
		for k, v in vars(self).items():
			s += "".join([" "] * offset)
			if isinstance(v, SimpleNamespace):
				s += f"{k}:\n{v.to_string(offset + 2)}"
			else:
				s += f"{k}: {v}\n"
		return s


def load_config() -> Config:
	config = Config(**{
		"paths": Config(**{
			"build": "",
			"samples": "",
			"autoannotation": "",
			"metadata": ""
		}),
		"params": Config(**{
			"k": 25,
			"k_pooling": 10,
			"n_factors": 96,
			"min_umis": 1500,
			"n_genes": 2000,
			"doublets_action": "remove",
			"doublets_method": "scrublet",
			"mask": ("cellcycle", "sex", "ieg", "mt"),
			"min_fraction_good_cells": 0.4,
			"skip_missing_samples": False,
			"clusterer": "louvain"  # or "surprise"
		}),
		"steps": ("doublets", "poisson_pooling", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "export"),
		"execution": Config(**{
			"n_cpus": 28,
			"n_gpus": 0,
			"memory": 128
		})
	})
	# Home directory
	f = os.path.join(os.path.abspath(str(Path.home())), ".cytograph")
	if os.path.exists(f):
		merge_config(config, f)
	# Set build folder
	if config.paths.build == "" or config.paths.build is None:
		config.paths.build = os.path.abspath(os.path.curdir)
	# Build folder
	f = os.path.join(config.paths.build, "config.yaml")
	if os.path.exists(f):
		merge_config(config, f)
	return config
