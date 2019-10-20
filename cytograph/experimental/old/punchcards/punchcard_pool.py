from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import development_mouse as dm
import luigi
from collections import defaultdict


def ixs_thatsort_a2b(a: np.ndarray, b: np.ndarray, check_content: bool=True) -> np.ndarray:
    "This is super duper magic sauce to make the order of one list to be like another"
    if check_content:
        assert len(np.intersect1d(a, b)) == len(a), f"The two arrays are not matching"
    return np.argsort(a)[np.argsort(np.argsort(b))]


class PunchcardPool(luigi.Task):  # Status: check the filter manager
    """
    Luigi Task to generate a particular slice of the data as specified by a punchcard

    `analysis` needs to match the name specified in the .yaml file in the folder ../punchcards
    """
    
    card = luigi.Parameter()
    punchcard_deck = dm.PunchcardParser()

    def requires(self) -> Iterator[luigi.Task]:
        """Parses the files in punchcard folder and returns required Tasks
        """
        punchcard_obj = self.punchcard_deck[self.card]
        return dm.parse_punchcard_require(punchcard_obj)

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(os.path.join(dm.paths().build, f"Pool_{self.card}.loom"))
        
    def run(self) -> None:
        analysis_obj = self.punchcard_deck[self.card]
        logging.debug(f"Generating the pooled file {self.card}.loom")
        
        with self.output().temporary_path() as out_file:
            dsout: loompy.LoomConnection = None
            # Try to drop the assumption that
            # clustering and the autoannotation are the i
            cluster_counter = 0
            reference_accession = None
            for input_dict in self.input():
                # NOTE: autoannotated is an Export Task but really an Aggragate task is enough, this is just for compatibility
                clustered, export_folder = input_dict[0], input_dict[1]
                autoannotated_fn = clustered.fn[:-5] + ".agg.loom"  # NOTE: this is some kind of workaround, points to the Aggragate task output
                logging.debug(f"Adding cells from the source file {clustered.fn}")
                ds = loompy.connect(clustered.fn, 'r')
                dsagg = loompy.connect(autoannotated_fn, 'r')
                
                # Select the tags as specified in the process file
                filter_bool = cg.FilterManager(analysis_obj, ds, dsagg, root=dm.paths().autoannotation).compute_filter()
                logging.debug(f"Plot the cell selection.")
                dm.plot_punchcard_selection(ds, os.path.join(export_folder.fn, f"Punchcard_{self.card}.png"), filter_bool)

                if reference_accession is None:
                    reference_accession = ds.row_attrs["Accession"]
                # NOTE: I don't know if the code below is updated
                order = ixs_thatsort_a2b(ds.row_attrs["Accession"], reference_accession)
                # NOTE: All of this would be much simpler using loompy2
                # NOTE: It should be substituted by scan
                for (ix, selection, vals) in ds.batch_scan_layers(axis=1, batch_size=dm.memory().axis1):
                    # Filter the cells that belong to the selected tags
                    subset = np.intersect1d(np.where(filter_bool)[0], selection)
                    if subset.shape[0] == 0:
                        continue
                    m = {}
                    for layer_name, chunk_of_matrix in vals.items():
                        m[layer_name] = vals[layer_name][order, :][:, subset - ix]
                    ca = {}
                    for key in ds.col_attrs:
                        if key == "Clusters":
                            # NOTE Special attention not to merge clusters
                            ca["Clusters_original"] = ds.col_attrs[key][subset]
                            ca[key] = ds.col_attrs[key][subset] + cluster_counter
                        else:
                            ca[key] = ds.col_attrs[key][subset]
                            
                    # Add data to the loom file
                    if dsout is None:
                        # create using main layer
                        loompy.create(out_file, m[""], ds.ra, ca)
                        dsout = loompy.connect(out_file)
                        # Add layers
                        for layer_name, chunk_of_matrix in m.items():
                            if layer_name == "":
                                continue
                            dsout[layer_name] = chunk_of_matrix
                    else:
                        dsout.add_columns(m, ca)
                # NOTE Special attention not to merge clusters
                renumbered_clusters = np.copy(dsout.col_attrs["Clusters"])
                _, renumbered_clusters[renumbered_clusters >= 0] = np.unique(renumbered_clusters[renumbered_clusters >= 0], return_inverse=True)
                dsout.set_attr("Clusters", renumbered_clusters, axis=1)
                cluster_counter = np.max(renumbered_clusters) + 1
