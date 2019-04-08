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
	feature_selection_method: str  # "variance", "markers"
	mask: List[str]


class Config(NamedTuple):
	paths: PathConfig
	params: ParamsConfig
	steps: List[str]  # "doublets", "poisson_pooling", "cells_qc", "batch_correction", "velocity", "nn", "embeddings", "clustering", "aggregate", "export"


config = Config()  # TODO: load the config from the build folder, users home etc. and merge them sensibly
