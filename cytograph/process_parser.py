from typing import *
import yaml
import os


class ProcessesParser(object):
	def __init__(self, root: str = "../dev-processes") -> None:
		self.root = root
		self._processes_dict = {}  # type: Dict
		self._load_defs()
	
	def _load_defs(self) -> None:
		for cur, dirs, files in os.walk(self.root):
			for file in files:
				if "yaml" in file:
					temp_dict = yaml.load(open(file))
					name = temp_dict["abbreviation"]
					self._processes_dict[name] = temp_dict

	@property
	def all_processes(self) -> List:
		return list(self._processes_dict.values())

	@property
	def all_processes_dict(self) -> Dict[str, Dict]:
		return dict(self._processes_dict)

	def __getitem__(self, key: Any) -> Dict:
		return self._processes_dict[key]
