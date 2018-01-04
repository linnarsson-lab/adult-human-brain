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
    def __init__(self, punchcard_obj: Dict, ds: loompy.LoomConnection, dsagg: loompy.LoomConnection=None) -> None:
        self.punchcard_obj = punchcard_obj
        self.ds = ds
        self.dsagg = dsagg
        # Read the autoannotation.aa.tab file and extract tags
        self.tags_per_cluster = list(self.dsagg.col_attrs["AutoAnnotation"])  # Previously cg.read_autoannotation()

    def _make_filter_aa(self, include_aa: List, exclude_aa: List) -> Tuple[np.ndarray, np.ndarray]:
        """Add and then remove clusters on the basis of the autoannotation
        """
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
                        logging.warning("Punchcards: include aa are not correctly fomratted")
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
                        logging.warning("Punchcards: exclude aa are not correctly formatted")
        in_aa = np.in1d(self.ds.col_attrs["Clusters"], list(selected_clusters))
        ex_aa = np.in1d(self.ds.col_attrs["Clusters"], list(deselected_clusters))
        return in_aa, ex_aa
    
    def make_filter_aa(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read the punchcard dictionary
        """
        include_aa = self.punchcard_obj["include"]["auto-annotations"]
        exclude_aa = self.punchcard_obj["exclude"]["auto-annotations"]
        in_aa, ex_aa = self._make_filter_aa(include_aa, exclude_aa)
        logging.debug("Filter Manager - autoannotation, include: %d  exclude:  %d" % (np.sum(in_aa), np.sum(ex_aa)))
        return in_aa, ex_aa
        
    def make_filter_classifier(self) -> Tuple[np.ndarray, np.ndarray]:
        include_class = self.punchcard_obj["include"]["classes"]
        exclude_class = self.punchcard_obj["exclude"]["classes"]
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
        logging.debug("Filter Manager - classifier, include: %d  exclude:  %d" % (np.sum(in_cla), np.sum(ex_cla)))
        return in_cla, ex_cla

    def make_filter_cluster(self) -> Tuple[np.ndarray, np.ndarray]:
        include_clust = self.punchcard_obj["include"]["clusters"]
        exclude_clust = self.punchcard_obj["exclude"]["clusters"]
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
        logging.debug("Filter Manager - cluster, include: %d  exclude:  %d" % (np.sum(in_clu), np.sum(ex_clu)))
        return in_clu, ex_clu

    def make_filter_category(self) -> Tuple[np.ndarray, np.ndarray]:
        aa = cg.AutoAnnotator.load_direct()
        categories_dict = defaultdict(list)  # type: DefaultDict
        for t in aa.tags:
            for c in t.categories:
                categories_dict[c].append(t.abbreviation)
        include_cat = self.punchcard_obj["include"]["categories"]
        exclude_cat = self.punchcard_obj["exclude"]["categories"]

        if include_cat == "all":
            include_aa = "all"  # type: Any
        else:
            include_aa = []
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
        
        if exclude_cat == "none":
            exclude_aa = "none"  # type: Any
        else:
            exclude_aa = []
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
        in_cat, ex_cat = self._make_filter_aa(include_aa, exclude_aa)
        logging.debug("Filter Manager - categories, include: %d  exclude:  %d" % (np.sum(in_cat), np.sum(ex_cat)))
        return in_cat, ex_cat

    def make_filter_time(self) -> np.ndarray:
        if self.punchcard_obj["timepoints"] == "all":
            return np.ones(self.ds.shape[1], dtype=bool)
        else:
            time_selected_int = [EP2int(i) for i in self.punchcard_obj["timepoints"]]
            in_time = np.in1d([EP2int(i) for i in self.ds.col_attrs["Age"]], time_selected_int)
            logging.debug("Filter Manager - time, include: %d " % (np.sum(in_time),))
            return in_time

    def make_filter_tissue(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.punchcard_obj["include"]["tissues"] == "all":
            in_tis = np.ones(self.ds.shape[1], dtype=bool)
        else:
            in_tis = np.in1d(self.ds.col_attrs["Tissue"], self.punchcard_obj["include"]["tissues"])
        if self.punchcard_obj["exclude"]["tissues"] == "none":
            ex_tis = np.zeros(self.ds.shape[1], dtype=bool)
        else:
            ex_tis = np.in1d(self.ds.col_attrs["Tissue"], self.punchcard_obj["exclude"]["tissues"])
        logging.debug("Filter Manager - tissue, include: %d  exclude:  %d" % (np.sum(in_tis), np.sum(ex_tis)))
        return in_tis, ex_tis

    def compute_filter(self) -> np.ndarray:
        in_aa, ex_aa = self.make_filter_aa()
        in_cat, ex_cat = self.make_filter_category()
        in_clu, ex_clu = self.make_filter_cluster()
        in_cla, ex_cla = self.make_filter_classifier()
        in_tis, ex_tis = self.make_filter_tissue()
        in_time = self.make_filter_time()
        filter_include = in_aa & in_cat & in_clu & in_cla & in_tis & in_time
        filter_exclude = (ex_aa | ex_cat | ex_clu | ex_cla | ex_tis)
        return filter_include & np.logical_not(filter_exclude)
