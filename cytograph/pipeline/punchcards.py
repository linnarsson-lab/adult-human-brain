from typing import *
from time import sleep
import os
import logging
import loompy
import yaml
import numpy as np
import luigi
from .cytograph2 import Cytograph2
from .aggregator import Aggregator
from .annotation import AutoAutoAnnotator
from .config import config
from .tasks import Pool


"""
RadialGlia:
	include: [Rgl, Rgl2, Rgl2c]
	onlyif: "(Age > 9) & (Age < 12) & (Tissue == 'Forebrain') & (~np.isin(Clusters, [2, 3, 4, 5])"
	steps: ["export"]
	params:
		k: 25
		k_pooling: 10
		n_genes: 500
		poisson_pooling: True
		velocity: True
		embedding: True
		doublets: "remove", "score", "none"
		min_umis: 1500

Neuroblast:
	include: [Nbl, NblF]

Other:
	include: True
	params:
		k_pooling: 10
"""


class PunchcardSubset:
	def __init__(self, name: str, card: Punchcard, include: List[str], onlyif: str, params: Dict[str, Any], steps: List[str]) -> None:
		self.name = name
		self.card = card
		self.include = include
		self.onlyif = onlyif
		self.params = params
		self.steps = steps

	def longname(self) -> str:
		name = self.name
		p: Optional[Punchcard] = self.card
		while p is not None:
			name = p.name + "_" + name
			p = p.parent
		return name


class Punchcard:
	def __init__(self, name: str, parent: Punchcard = None, subsets: List[PunchcardSubset] = None, children: Dict[str, Punchcard] = None) -> None:
		self.name = name
		self.parent = parent
		self.subsets: Dict[str, PunchcardSubset] = {}
		if subsets is not None:
			self.subsets = {s.name: s for s in subsets}
		self.children: Dict[str, Punchcard] = {}
		if children is not None:
			self.children = children

	@staticmethod
	def load_recursive(path: str, parent: Punchcard = None) -> Punchcard:
		name = os.path.basename(path).split(".")[0]
		with open(path) as f:
			spec = yaml.load(f)
		p = Punchcard(name, parent, None, None)
		subsets = []
		for s, items in spec.items():
			subsets.append(PunchcardSubset(s, p, items.get("include"), items.get("onlyif"), items.get("params"), items.get("steps")))
		p.subsets = {s.name: s for s in subsets}

		p.children = {}
		for s in subsets:
			subset_path = os.path.join(os.path.splitext(path)[0], "_", s.name, ".yaml")
			if os.path.exists(subset_path):
				p.children[s.name] = Punchcard.load_recursive(subset_path, p)
		return p

	def get_leaves(self) -> List[PunchcardSubset]:
		assert(self.subsets is not None)
		assert(self.children is not None)
		result = [s for s in self.subsets.values() if s.name not in self.children]
		for c in self.children.values():
			result += c.get_leaves()
		return result
	

class PunchcardDeck:
	def __init__(self, path: str) -> None:
		self.path = path
		self.root = Punchcard.load_recursive(os.path.join(path, "Root.yaml"), None)

	def task(self) -> Pool:
		return Pool(subsets=self.root.get_leaves())
