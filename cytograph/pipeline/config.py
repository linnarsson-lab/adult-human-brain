from typing import *
import luigi


class PathConfig(NamedTuple):
	build: str
	samples: str
	metadata: str
	autoannotation: str


class ParamsConfig(NamedTuple):
	k: int
	k_pooling: int
	n_factors: int
	min_umis: int
	n_genes: int
	doublets_action: str  # "score", "remove"
	doublets_method: str  # "scrublet", "doublet_finder"
	mask: List[str]


class ExecutionConfig:
	max_jobs: int
	local: bool
	n_cpus: int
	memory: int


class Config(NamedTuple):
	paths: PathConfig
	params: ParamsConfig
	steps: List[str]
	execution: ExecutionConfig


#
# Load configs from the following locations in this order (merging each):
#
# 	1. The builtin defaults
#   2. The user's home directory
#   3. The directory above the current directory
#   4. The current directory (& set this to be the build folder)
#
config = Config(
	paths=PathConfig(
		build="",
		samples="",
		metadata="",
		autoannotation=""
	),
	params=ParamsConfig(
		k=25,
		k_pooling=10,
		n_factors=96,
		min_umis=1500,
		n_genes=2000,
		doublets_action="remove",
		doublets_method="scrublet",
		mask=["cellcycle", "sex", "ieg"]
	),
	steps=["doublets", "poisson_pooling", "cells_qc", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "export"]
)
