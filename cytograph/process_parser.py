from typing import *
import yaml
import os


class ProcessesParser(object):
	def __init__(self, root: str = "../dev-processes") -> None:
		self.root = root
		self._processes_dict = {}  # type: Dict
		self.model = {}  # type: Dict
		self._load_model()
		self._load_defs()

	def _load_model(self) -> None:
		self.model = yaml.load(open(os.path.join(self.root, "Model.yaml")))

	def _load_defs(self) -> None:
		for cur, dirs, files in os.walk(self.root):
			for file in files:
				if "yaml" in file:
					temp_dict = yaml.load(open(file))
					name = temp_dict["abbreviation"]
					model_copy = dict(self.model)

					# Do an update of the model dictionary, so to keep the defaults
					for k, v in self.model.items():
						if type(v) == dict:
							for kk, vv in v.items():
								if type(vv) == dict:
									for kkk, vvv in vv.items():
										try:
											model_copy[k][kk][kkk] = temp_dict[k][kk][kkk]
										except ValueError:
											pass
								else:
									try:
										model_copy[k][kk] = temp_dict[k][kk]
									except ValueError:
										pass
						else:
							try:
								model_copy[k] = temp_dict[k]
							except ValueError:
								pass
					self._processes_dict[name] = model_copy

	@property
	def all_processes(self) -> List:
		return list(self._processes_dict.values())

	@property
	def all_processes_dict(self) -> Dict[str, Dict]:
		return dict(self._processes_dict)

	def __getitem__(self, key: Any) -> Dict:
		return self._processes_dict[key]
