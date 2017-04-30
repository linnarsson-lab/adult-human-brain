from typing import *
import cytograph as cg
import loompy
import numpy as np
import logging
from collections import defaultdict


def EP2int(timepoint: str) -> int:
    if "P" in timepoint:
        return int(float(timepoint.lstrip("P"))) + 19
    else:
        return int(float(timepoint.lstrip("E")))


class FilterManager(object):
    def __init__(self, process_obj: Dict, ds: loompy.LoomConnection, aa_file_name: str=None) -> None:
        self.process_obj = process_obj
        self.ds = ds
        self.aa_file_name = aa_file_name
        # Read the autoannotation.aa.tab file and extract tags
        self.tags_per_cluster = cg.read_autoannotation(self.aa_file_name)

    def _make_filter_aa(self, include_aa: List, exclude_aa: List) -> Tuple[np.ndarray, np.ndarray]:
        # Add and then remove cluster on the basis of the autoannotation
        selected_clusters = set()  # type: set
        deselected_clusters = set()  # type: set
        for cluster_ix, tags in enumerate(self.tags_per_cluster):
            # Deal with the inclusions
            if include_aa == "all":
                selected_clusters = set(list(range(len(self.tags_per_cluster))))
            else:
                for include_entry in include_aa:
                    if type(include_entry) == list:
                        if np.alltrue(np.in1d(include_entry, tags)):
                            selected_clusters |= {cluster_ix}
                    elif type(include_entry) == str:
                        if include_entry in tags:
                            selected_clusters |= {cluster_ix}
                    else:
                        logging.warning("Processes: include aa are not correctly fomratted")
            # Deal with the exclusions
            if exclude_aa == "none":
                deselected_clusters = set()
            else:
                for exclude_entry in exclude_aa:
                    if type(exclude_entry) == list:
                        if np.alltrue(np.in1d(exclude_entry, tags)):
                            deselected_clusters |= {cluster_ix}
                    elif type(exclude_entry) == str:
                        if include_entry in tags:
                            deselected_clusters |= {cluster_ix}
                    else:
                        logging.warning("Processes: exclude aa are not correctly fomratted")
        in_aa = np.in1d(self.ds.col_attrs["Clusters"], list(selected_clusters))
        ex_aa = np.in1d(self.ds.col_attrs["Clusters"], list(deselected_clusters))
        return in_aa, ex_aa
    
    def make_filter_aa(self) -> Tuple[np.ndarray, np.ndarray]:
        # Read the process dictionary
        include_aa = self.process_obj["include"]["auto-annotations"]
        exclude_aa = self.process_obj["exclude"]["auto-annotations"]
        return self._make_filter_aa(include_aa, exclude_aa)
        
    def make_filter_classifier(self) -> Tuple[np.ndarray, np.ndarray]:
        include_class = self.process_obj["include"]["classes"]
        exclude_class = self.process_obj["exclude"]["classes"]
        in_cla = np.zeros(self.ds.shape[1], dtype=bool)
        ex_cla = np.zeros(self.ds.shape[1], dtype=bool)
        # Deals with inclusions
        if include_class == "all":
            in_cla = np.ones(self.ds.shape[1], dtype=bool)
        else:
            for cl in include_class:
                in_cla |= self.ds.col_attrs["Class_%s" % cl.title()] > 0.5
        # Deals with exclusions
        if exclude_class == "none":
            pass
        else:
            for cl in exclude_class:
                ex_cla |= self.ds.col_attrs["Class_%s" % cl.title()] > 0.5
        return in_cla, ex_cla

    def make_filter_cluster(self) -> Tuple[np.ndarray, np.ndarray]:
        include_clust = self.process_obj["include"]["clusters"]
        exclude_clust = self.process_obj["exclude"]["clusters"]
        # Deals with inclusions
        if include_clust == "all":
            in_clu = np.ones(self.ds.shape[1], dtype=bool)
        else:
            in_clu = np.in1d(self.ds.col_attrs["Clusters"], include_clust)
        # Deals with exclusions
        if exclude_clust == "none":
            ex_clu = np.zeros(self.ds.shape[1], dtype=bool)
        else:
            ex_clu = np.in1d(self.ds.col_attrs["Clusters"], exclude_clust)
        return in_clu, ex_clu

    def make_filter_category(self) -> Tuple[np.ndarray, np.ndarray]:
        aa = cg.AutoAnnotator()
        aa.load_defs()
        categories_dict = defaultdict(list)  # type: DefaultDict
        for t in aa.tags:
            for c in t.categories:
                categories_dict[c].append(t.abbreviation)
        include_cat = self.process_obj["include"]["categories"]
        exclude_cat = self.process_obj["exclude"]["categories"]

        include_aa = []  # type: list
        for cat in include_cat:
            if type(cat) == str:
                include_aa += categories_dict[cat]
            elif type(cat) == list:
                intersection = set(categories_dict[cat[0]])
                for c in cat[1:]:
                    intersection &= set(categories_dict[c])
                include_aa += list(intersection)
            else:
                logging.warning("Processes: exclude categories are not correctly formatted")
        
        exclude_aa = []  # type: list
        for cat in exclude_cat:
            if type(cat) == str:
                exclude_aa += categories_dict[cat]
            elif type(cat) == list:
                intersection = set(categories_dict[cat[0]])
                for c in cat[1:]:
                    intersection &= set(categories_dict[c])
                exclude_aa += list(intersection)
            else:
                logging.warning("Processes: exclude categories are not correctly formatted")
        return self._make_filter_aa(include_aa, exclude_aa)

    def make_filter_time(self) -> np.ndarray:
        if self.process_obj["timepoints"] == "all":
            return np.ones(self.ds.shape[1], dtype=bool)
        else:
            time_selected_int = [EP2int(i) for i in self.process_obj["timepoints"]]
            in_time = np.in1d([EP2int(i) for i in self.ds.col_attrs["Age"]], time_selected_int)
            return in_time

    def compute_filter(self) -> np.ndarray:
        in_aa, ex_aa = self.make_filter_aa()
        in_cat, ex_cat = self.make_filter_category()
        in_clu, ex_clu = self.make_filter_cluster()
        in_cla, ex_cla = self.make_filter_classifier()
        in_time = self.make_filter_time()
        filter_include = (in_aa | in_cat | in_clu | in_cla) & in_time
        filter_exclude = (ex_aa | ex_cat | ex_clu | ex_cla)
        return filter_include & np.logical_not(filter_exclude)
