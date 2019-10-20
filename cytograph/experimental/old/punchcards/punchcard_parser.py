from typing import *
import yaml
import os
import luigi
import cytograph as cg
import numpy as np
import development_mouse as dm
import logging
from scipy import sparse
from collections import defaultdict
import copy

# those are the analyses allowed, if a kind of analysis is not here cannot be run using the punchcard submodule
# require_type_dict = {"AggregatePunchcard": dm.AggregatePunchcard,
#                      "ClusterPunchcard": dm.ClusterPunchcard,
#                      "ClusterL1": dm.ClusterL1,
#                      "ExportPunchcard": dm.ExportPunchcard,
#                      "ExportL1": dm.ExportL1,
#                      "PunchcardPool": dm.PunchcardPool,
#                      "EstimateVelocityPunchcard": dm.EstimateVelocityPunchcard,
#                      "VisualizeVelocityPunchcard": dm.VisualizeVelocityPunchcard,
#                      "Level1": dm.Level1}


class PunchcardParser(object):  # Status: needs to be run but looks ok
    def __init__(self, root: str = dm.paths().punchcards) -> None:
        # NOTE: root should be changed to a directory defined in a ~/.cytograph_rc file
        self.root = os.path.abspath(root)
        self._punchcard_dict = {}  # type: Dict
        self.model = {}  # type: Dict
        self._load_model()
        self._load_defs()
        self._has_printed: Set[str] = set()

    def _load_model(self) -> None:
        self.model = yaml.load(open(os.path.join(self.root, "Model.yaml")))

    def _load_defs(self) -> None:
        debug_msgs = defaultdict(list)  # type: dict
        for cur, dirs, files in os.walk(self.root):
            if any(i in cur.split("/") for i in ["ignore", "exclude", "old", ".git"]):
                continue
            for file in files:
                if ((".yaml" == file[-5:]) or (".yml" == file[-4:])) and ("Model.yaml" not in file):
                    logging.debug(f"Reading {os.path.join(cur, file)}")
                    temp_dict = yaml.load(open(os.path.join(cur, file)))
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
                                            debug_msgs[name].append("Punchcard %s `%s:%s:%s` was not found. The Default `%s` will be used" % (name, k, kk, kkk, model_copy[k][kk][kkk]))
                                else:
                                    try:
                                        model_copy[k][kk] = temp_dict[k][kk]
                                    except KeyError:
                                        debug_msgs[name].append("Punchcard %s `%s:%s` was not found. The Default `%s` will be used" % (name, k, kk, model_copy[k][kk]))
                        else:
                            try:
                                model_copy[k] = temp_dict[k]
                            except KeyError:
                                debug_msgs[name].append("Punchcard %s `%s` was not found. The Default `%s` will be used" % (name, k, model_copy[k]))
                    self._punchcard_dict[name] = copy.deepcopy(model_copy)
        self.debug_msgs = debug_msgs

    @property
    def all_analyses(self) -> List:
        return list(self._punchcard_dict.values())

    @property
    def all_punchcard_dict(self) -> Dict[str, Dict]:
        return dict(self._punchcard_dict)

    def __getitem__(self, key: Any) -> Dict:
        if not (key in self._has_printed):
            for i in self.debug_msgs[key]:
                logging.debug(i)
            self._has_printed.add(key)
        return self._punchcard_dict[key]

    def prune_leaves(self) -> List[str]:
        punchcard_names = np.array(list(self._punchcard_dict.keys()))
        n = punchcard_names.shape[0]
        graph = sparse.lil_matrix((n, n), dtype=bool)
        for i, key in enumerate(punchcard_names):
            # Only one parent is allowed, to avoid redundances
            tasks = self._punchcard_dict[key]["require"]
            if len(tasks) > 1 or tasks[0]["type"] != "Punchcard":
                continue
            task = tasks[0]
            j = np.where(punchcard_names == task["kwargs"]["card"])[0][0]
            graph[i, j] = True
        # NOTE: I might have to deal with special case where I want to consider as a leaf a direct dependency from L1
        dont_have_child = graph.sum(0).A.flat[:] == 0
        have_parent = graph.sum(1).A.flat[:] > 0
        leaves_ixs = np.where(dont_have_child & have_parent)[0]
        leaves = list(punchcard_names[leaves_ixs])
        names_str = '\n'.join(leaves)
        logging.info(f"Pruned leaves:\n{names_str}")
        return leaves


def parse_punchcard_require(punchcard_obj: Dict) -> List[luigi.Task]:
    """Takes a dictionary parsed from the yaml file and returns the correspnding list of Tasks requirement

    The current version assumes that the input wil always be a WrapperTask
    """
    requirements: List[luigi.Task] = []
    c = 0
    for i in range(len(punchcard_obj["require"])):
        requirement_entry = punchcard_obj["require"][i]
        requirement_type = requirement_entry["type"]
        requirement_kwargs = requirement_entry["kwargs"]
        Task = getattr(dm, requirement_type)
        if requirement_type == 'Level1Analysis':
            try:
                tissues = cg.PoolSpec(dm.paths().poolspec).tissues_for_project(requirement_kwargs["project"])
            except KeyError:
                tissues = requirement_kwargs['tissue']
            for tissue in tissues:
                requirements.append([dm.ClusterL1(tissue=tissue), dm.ExportL1(tissue=tissue)])
        elif issubclass(Task, luigi.WrapperTask):
            requirements.append(Task(**requirement_kwargs).requires())
        else:
            assert issubclass(Task, luigi.Task), f"{requirement_type} is not valid Task name"
            if c == 0:
                requirements.append([Task(**requirement_kwargs)])
                c += 1
            else:
                requirements[-1].append(Task(**requirement_kwargs))
    return requirements


def parse_punchcard_run(punchcard_obj: Dict) -> Iterator[luigi.Task]:
    """Yields luigi.Tasks after parsing out a dictionary describing the kind of tasks and their arguments
    """
    # the following safenames is implemented to make the gettattr statement secure
    safenames = set()  # type: set
    for k, v in dm.__dict__.items():
        if type(v) == luigi.task_register.Register:
            safenames |= {k}
    if punchcard_obj["run"] is None:
        return None
    for task2run in punchcard_obj["run"]:
        task_type = task2run["type"]
        if "kwargs" in task2run:
            task_kwargs = task2run["kwargs"]
        else:
            task_kwargs = {}
        
        Task_class = getattr(dm, task_type)  # Since I don't use eval anymore this is basically safe

        assert issubclass(Task_class, luigi.Task) or issubclass(Task_class, luigi.WrapperTask), f"{task_type} is not valid Task name"

        def Task(card: Any) -> luigi.Task:
            d = {"card": card}
            d.update(task_kwargs)
            return Task_class(**d)
        
        yield Task
