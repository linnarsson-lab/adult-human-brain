import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Union

import yaml


class Punchcard:
	def __init__(self, name: str, parent: "Punchcard" = None, subsets: List["PunchcardSubset"] = None, children: Dict[str, "Punchcard"] = None) -> None:
		self.name = name
		self.parent = parent
		self.subsets: Dict[str, PunchcardSubset] = {}
		if subsets is not None:
			self.subsets = {s.name: s for s in subsets}
		self.children: Dict[str, Punchcard] = {}
		if children is not None:
			self.children = children

	@staticmethod
	def load_recursive(path: str, parent: "Punchcard" = None) -> "Punchcard":
		name = os.path.basename(path).split(".")[0]
		if not os.path.exists(path):
			logging.error(f"Punchcard {path} not found.")
			sys.exit(1)
		with open(path) as f:
			spec = yaml.load(f)
		p = Punchcard(name, parent, None, None)
		subsets = []
		logging.debug(f"Loading punchcard spec for {name}")
		for s, items in spec.items():
			if not re.match("[A-Za-z0-9]+", s):
				logging.error(f"Subset names can only contain letters and numbers, and '{s}' is therefore invalid")
				sys.exit(1)
			subsets.append(PunchcardSubset(s, p, items.get("include"), items.get("onlyif"), items.get("params"), items.get("steps"), items.get("execution")))
		p.subsets = {s.name: s for s in subsets}

		p.children = {}
		for s in subsets:
			if name == "Root":
				subset_path = os.path.splitext(path)[0][:-4] + s.name + ".yaml"
			else:
				subset_path = os.path.splitext(path)[0] + "_" + s.name + ".yaml"
			if os.path.exists(subset_path):
				p.children[s.name] = Punchcard.load_recursive(subset_path, p)
		return p

	def get_leaves(self) -> List["PunchcardSubset"]:
		assert(self.subsets is not None)
		assert(self.children is not None)
		result = [s for s in self.subsets.values() if s.name not in self.children]
		for c in self.children.values():
			result += c.get_leaves()
		return result


class PunchcardSubset:
	def __init__(self, name: str, card: Punchcard, include: Union[List[str], List[List[str]]], onlyif: str, params: Dict[str, Any], steps: List[str], execution: Dict[str, Any]) -> None:
		if name == "Pool" or name == "View":
			raise ValueError(f"Subset '{name}' in punchcard '{card.name}' not allowed ('Pool' and 'View' are reserved names).")
		self.name = name
		self.card = card
		self.include = include
		self.onlyif = onlyif
		self.params = params
		self.steps = steps
		self.execution = execution

	def longname(self) -> str:
		if self.card.name == "Root":
			return self.name
		else:
			return self.card.name + "_" + self.name

	def dependency(self) -> Optional[str]:
		names = self.longname().split("_")
		if len(names) == 1:
			return None
		return "_".join(names[:-1])


class PunchcardView:
	def __init__(self, path: str) -> None:
		self.name = os.path.basename(path).split(".")[0]
		if not os.path.exists(path):
			logging.error(f"View {path} not found.")
			sys.exit(1)
		with open(path) as f:
			spec = yaml.load(f)
		self.steps = spec.get("steps")
		self.sources = spec.get("sources")
		self.include = spec.get("include")
		self.onlyif = spec.get("onlyif")
		self.params = spec.get("params")
		self.execution = spec.get("execution")

	@staticmethod
	def load_all(path: str) -> List["PunchcardView"]:
		result: List[PunchcardView] = []
		if os.path.exists(path):
			for f in os.listdir(path):
				if f.lower().endswith(".yaml"):
					result.append(PunchcardView(os.path.join(path, f)))
		return result


class PunchcardDeck:
	def __init__(self, build_path: str) -> None:
		self.path = build_path
		self.root = Punchcard.load_recursive(os.path.join(build_path, "punchcards", "Root.yaml"), None)
		self.views = PunchcardView.load_all(os.path.join(build_path, "views"))

		# Check the samples specifications, and make sure they make sense
		for subset in self.root.subsets.values():
			if not isinstance(subset.include, list):
				logging.error(f"Error in '{subset.longname()}'; every 'include' in Root.yaml must be a non-empty list of lists of samples.")
				sys.exit(1)
			for sample in subset.include:
				if not isinstance(sample, list):
					logging.error(f"Error in '{subset.longname()}'; every 'include' in Root.yaml must be a non-empty list of lists of samples.")
					sys.exit(1)
			if subset.onlyif != "" and subset.onlyif is not None:
				logging.error(f"Error in '{subset.longname()}'; 'onlyif' clauses cannot be used in Root.yaml, only in downstream punchcards.")
				sys.exit(1)

	def get_leaves(self) -> List[PunchcardSubset]:
		return self.root.get_leaves()

	def _get_subset(self, card: Punchcard, name: str) -> Optional[PunchcardSubset]:
		for s in card.subsets.values():
			if s.longname() == name:
				return s
		for c in card.children.values():
			s = self._get_subset(c, name)  # type: ignore
			if s is not None:
				return s
		return None

	def get_subset(self, name: str) -> Optional[PunchcardSubset]:
		return self._get_subset(self.root, name)

	def _get_card(self, card: Punchcard, name: str) -> Optional[Punchcard]:
		if name == card.name:
			return card
		for c in card.children.values():
			temp = self._get_card(c, name)
			if temp is not None:
				return temp
		return None

	def get_card(self, name: str) -> Optional[Punchcard]:
		return self._get_card(self.root, name)

	def get_view(self, name: str) -> Optional[PunchcardView]:
		for v in self.views:
			if v.name == name:
				return v
		return None
