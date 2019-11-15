import os
from pathlib import Path
from typing import Union, Optional
from types import SimpleNamespace
from .punchcards import PunchcardSubset, PunchcardView

import yaml


def merge_namespaces(a: SimpleNamespace, b: SimpleNamespace) -> None:
	for k, v in vars(b).items():
		if isinstance(v, SimpleNamespace):
			merge_namespaces(a.__dict__[k], v)
		else:
			a.__dict__[k] = v


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

	def merge_with(self, path: str) -> None:
		if not os.path.exists(path):
			raise IOError(f"Config path {path} not found.")

		with open(path) as f:
			defs = yaml.load(f, Loader=yaml.Loader)

		if "paths" in defs:
			merge_namespaces(self.paths, SimpleNamespace(**defs["paths"]))
		if "params" in defs:
			merge_namespaces(self.params, SimpleNamespace(**defs["params"]))
		if "steps" in defs:
			self.steps = defs["steps"]
		if "execution" in defs:
			merge_namespaces(self.execution, SimpleNamespace(**defs["execution"]))


def load_config(subset_obj: Union[Optional[PunchcardSubset], Optional[PunchcardView]] = None) -> Config:
	config = Config(**{
		"paths": Config(**{
			"build": "",
			"samples": "",
			"autoannotation": "",
			"metadata": "",
			"fastqs": "",
			"index": ""
		}),
		"params": Config(**{
			"k": 25,
			"k_pooling": 10,
			"factorization": "HPF",
			"n_factors": 96,
			"min_umis": 1500,
			"n_genes": 2000,
			"doublets_action": "remove",
			"doublets_method": "doubletFinder",
			"mask": ("cellcycle", "sex", "ieg", "mt"),
			"min_fraction_good_cells": 0.4,
			"max_fraction_MT_genes":0.05,
			"min_fraction_unspliced_reads": 0.2,
			"min_fraction_genes_UMI": 0.3,
			"max_doubletFinder_TH": 0.4,
			"skip_missing_samples": False,
			"features": "enrichment", # or "variance"
			"passedQC": False,
			"clusterer": "louvain"  # or "surprise"
		}),
		"steps": ("doublets", "poisson_pooling", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "skeletonize", "export"),
		"execution": Config(**{
			"n_cpus": 4,
			"n_gpus": 0,
			"memory": 128
		})
	})
	# Home directory
	f = os.path.join(os.path.abspath(str(Path.home())), ".cytograph")
	if os.path.exists(f):
		config.merge_with(f)
	# Set build folder
	if config.paths.build == "" or config.paths.build is None:
		config.paths.build = os.path.abspath(os.path.curdir)
	# Build folder
	f = os.path.join(config.paths.build, "config.yaml")
	if os.path.exists(f):
		config.merge_with(f)
	# Current subset or view
	if subset_obj is not None:
		if subset_obj.params is not None:
			merge_namespaces(config.params, SimpleNamespace(**subset_obj.params))
		if subset_obj.steps != [] and subset_obj.steps is not None:
			config.steps = subset_obj.steps
		if subset_obj.execution is not None:
			merge_namespaces(config.execution, SimpleNamespace(**subset_obj.execution))

	return config
