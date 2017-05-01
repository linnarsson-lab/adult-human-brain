from typing import *
import yaml
import os
import luigi
import cytograph as cg


analysis_type_dict = {"Level1": luigi.Level1, "SudyProcess": luigi.StudyProcess}


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
				if ".yaml" in file or ".yml" in file:
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


def parse_project_requirements(process_obj: Dict) -> List[Tuple[luigi.Task]]:
	"""
	This assume the requirements be always a TaskWrapper
	"""
	requirements = []  # type: List[luigi.WrapperTask]
	for i in range(len(process_obj["parent_analyses"])):
		parent_type = process_obj["parent_analyses"][i]["type"]
		parent_kwargs = process_obj["parent_analyses"][i]["kwargs"]
		if parent_type not in analysis_type_dict:
			raise NotImplementedError("type: %s not allowed, you need to allow it adding it to analysis_type_dict" % parent_type)
		Analysis = analysis_type_dict[parent_type]
		if parent_kwargs == {}:
			requirements += list(Analysis().requires())
		else:
			requirements += list(Analysis(**parent_kwargs).requires())
	return requirements


def parse_project_todo(process_obj: Dict) -> Iterator[luigi.Task]:
	for analysis_entry in process_obj["todo_analyses"]:
		analysis_type, analysis_kwargs = analysis_entry["type"], analysis_entry["kwargs"]
		if analysis_type not in analysis_type_dict:
			raise NotImplementedError("type: %s not allowed, you need to allow it adding it to analysis_type_dict" % analysis_type)
		else:
			Analysis_class = analysis_type_dict[analysis_type]

			def Analysis(x: Any) -> luigi.Task:
				return Analysis_class(x, **analysis_kwargs)
			
			yield Analysis
