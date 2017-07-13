from typing import *
import yaml
import os
import luigi
import cytograph as cg
from .luigi import Level1, PerformAnalysis, Level1Analysis
import logging
from collections import defaultdict
import copy

# those are the analyses allowed, if a kind of analysis is not here cannot be run using the analysis submodule
analysis_type_dict = {"Level1": Level1, "PerformAnalysis": PerformAnalysis}


class AnalysesParser(object):  # Status: needs to be run but looks ok
    def __init__(self, root: str = "../cg-analysis") -> None:
        self.root = root
        self._analyses_dict = {}  # type: Dict
        self.model = {}  # type: Dict
        self._load_model()
        self._load_defs()

    def _load_model(self) -> None:
        self.model = yaml.load(open(os.path.join(self.root, "Model.yaml")))

    def _load_defs(self) -> None:
        debug_msgs = defaultdict(list)  # type: dict
        for cur, dirs, files in os.walk(self.root):
            for file in files:
                if ((".yaml" in file) or (".yml" in file)) and ("Model.yaml" not in file):
                    temp_dict = yaml.load(open(os.path.join(self.root, file)))
                    name = temp_dict["abbreviation"]
                    model_copy = copy.deepcopy(self.model)

                    # Do an update of the model dictionary, so to keep the defaults
                    for k, v in self.model.items():
                        if type(v) == dict:
                            for kk, vv in v.items():
                                if type(vv) == dict:
                                    for kkk, vvv in vv.items():
                                        try:
                                            model_copy[k][kk][kkk] = temp_dict[k][kk][kkk]
                                        except KeyError:
                                            debug_msgs[name].append("Analysis %s `%s:%s:%s` was not found. The Default `%s` will be used" % (name, k, kk, kkk, model_copy[k][kk][kkk]))
                                else:
                                    try:
                                        model_copy[k][kk] = temp_dict[k][kk]
                                    except KeyError:
                                        debug_msgs[name].append("Analysis %s `%s:%s` was not found. The Default `%s` will be used" % (name, k, kk, model_copy[k][kk]))
                        else:
                            try:
                                model_copy[k] = temp_dict[k]
                            except KeyError:
                                debug_msgs[name].append("Analysis %s `%s` was not found. The Default `%s` will be used" % (name, k, model_copy[k]))
                    self._analyses_dict[name] = copy.deepcopy(model_copy)
                    self.debug_msgs = debug_msgs

    @property
    def all_analyses(self) -> List:
        return list(self._analyses_dict.values())

    @property
    def all_analyses_dict(self) -> Dict[str, Dict]:
        return dict(self._analyses_dict)

    def __getitem__(self, key: Any) -> Dict:
        for i in self.debug_msgs[key]:
            logging.debug(i)
        return self._analyses_dict[key]


def parse_analysis_requirements(process_obj: Dict) -> List[Tuple[luigi.Task]]:
    """
    This assume the requirements be always a TaskWrapper
    """
    requirements: List[luigi.WrapperTask] = []
    for i in range(len(process_obj["parent_analyses"])):
        parent_type = process_obj["parent_analyses"][i]["type"]
        parent_kwargs = process_obj["parent_analyses"][i]["kwargs"]
        if parent_type not in analysis_type_dict:
            raise NotImplementedError("type: %s not allowed, you need to allow it adding it to analysis_type_dict" % parent_type)
        Analysis = analysis_type_dict[parent_type]
        if parent_kwargs == {}:  # maybe there is no need of this if statement
            requirements += list(Analysis().requires())  # Requires returns an Iterator
        else:
            requirements += list(Analysis(**parent_kwargs).requires())
    return requirements


def parse_analysis_todo(process_obj: Dict) -> Iterator[luigi.Task]:
    """Yields luigi.Tasks after parsing out a dictionary describing the kind of tasks and their arguments
    """
    # the following safenames is implemented to make the eval statement secure
    safenames = set()  # type: set
    for k, v in cg.__dict__.items():
        if type(v) == luigi.task_register.Register:
            safenames |= {k}
    for analysis_entry in process_obj["todo_analyses"]:
        analysis_type, analysis_kwargs = analysis_entry["type"], analysis_entry["kwargs"]
        if analysis_type not in safenames:
            raise NotImplementedError("type: %s not allowed, becouse is not a valid luigi task" % analysis_type)
        else:
            Analysis_class = getattr(cg, analysis_type)  # eval("cg.%s" % analysis_type)

            def Analysis(analysis: Any) -> luigi.Task:
                return Analysis_class(analysis, **analysis_kwargs)
            
            yield Analysis
