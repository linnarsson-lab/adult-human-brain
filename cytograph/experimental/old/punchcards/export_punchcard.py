from typing import *
import os
import logging
import loompy
from scipy import sparse
import numpy as np
import networkx as nx
import cytograph as cg
import development_mouse as dm
import luigi


class ExportPunchcard(luigi.Task):
    """
    Luigi Task to export summary files for a punchcard analysis
    """
    card = luigi.Parameter(description="Name of punchcard")
    n_markers = luigi.IntParameter(default=10, description="number of markers to export")

    def requires(self) -> List[luigi.Task]:
        # NOTE before the order was AggregatePunchcard, ClusterPunchcard but did not make sense
        return [dm.ClusterPunchcard(card=self.card),
                dm.AggregatePunchcard(card=self.card)]

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(os.path.join(dm.paths().build, f"{self.card}_exported"))

    def run(self) -> None:
        logging = cg.logging(self, True)
        logging.info("Exporting cluster data")
        with self.output().temporary_path() as out_dir:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            dsagg = loompy.connect(self.input()[1].fn)
            dsagg.export(os.path.join(out_dir, f"{self.card}_expression.tab"))
            dsagg.export(os.path.join(out_dir, f"{self.card}_enrichment.tab"), layer="enrichment")
            # dsagg.export(os.path.join(out_dir, f"{self.card}_enrichment_q.tab"), layer="enrichment_q")
            dsagg.export(os.path.join(out_dir, f"{self.card}_trinaries.tab"), layer="trinaries")

            ds = loompy.connect(self.input()[0].fn)

            logging.info("Plotting manifold graph with auto-annotation")
            tags = list(dsagg.col_attrs["AutoAnnotation"])
            cg.plot_graph(ds, os.path.join(out_dir, f"L1_{self.card}_manifold.aa.png"), tags)

            try:
                logging.info("Plotting manifold graph with auto-annotation, colored by age")
                cg.plot_graph_age(ds, os.path.join(out_dir, f"L1_{self.card}_manifold.age.png"), tags)
            except:
                pass

            try:
                logging.info("Plotting abstracted graph with auto-annotation")
                dm.plot_abs_graph(ds, dsagg, os.path.join(out_dir, f"L1_{self.card}_absgraph.aa.png"), tags)
            except:
                pass

            logging.info("Plotting manifold graph with auto-auto-annotation")
            tags = list(dsagg.col_attrs["MarkerGenes"])
            cg.plot_graph(ds, os.path.join(out_dir, f"L1_{self.card}_manifold.aaa.png"), tags)

            logging.info("Plotting marker heatmap")
            cg.plot_markerheatmap(ds, dsagg, n_markers_per_cluster=self.n_markers, out_file=os.path.join(out_dir, f"L1_{self.card}_heatmap.pdf"))

            try:
                logging.info("Plotting quality class on t-SNE")
                tags = list(dsagg.col_attrs["AutoAnnotation"])
                cluster_mapping = {int(i.split(":")[0]): i.split(":")[1] for i in open(self.input()["NameQualityClusters"].fn).read().rstrip().split()}
                dm.plot_quality_graph(ds, dsagg, out_file=os.path.join(out_dir, f"L1_{self.card}_quality_tsne.png"),
                                    cluster_mapping=cluster_mapping, tags=tags)
                logging.info("Plotting quality class in pie chart")
                plt.figure(None, (10, 10))
                labels = ds.col_attrs["QualityClass"].astype(int)
                unique, counts = np.unique(labels, return_counts=True)
                labelnames = [cluster_mapping[ix] for ix in unique]
                patches, texts = plt.pie(counts)
                plt.legend(patches, labelnames, bbox_to_anchor=(0.1, 1), fontsize=15)
                plt.savefig(os.path.join(out_dir, "L1_" + self.card + "_quality_pie.png"))
            except:
                pass

            # cytograph2 plots
            try:
                logging.info("Plotting UMAP")
                cg.plot_graph(ds, os.path.join(out_dir, f"L1_{self.card}_UMAP_manifold.aaa.png"), tags, embedding="UMAP")
                logging.info("Plotting UMI and gene counts")
                cg.plot_umi_genes(ds, out_file=os.path.join(out_dir, "L1_" + self.card + "_umi_genes.png"))
                logging.info("Plotting factors")
                cg.plot_factors(ds, base_name=os.path.join(out_dir, "L1_" + self.card + "_factors"))
                logging.info("Plotting cell cycle")
                cg.plot_cellcycle(ds, out_file=os.path.join(out_dir, "L1_" + self.card + "_cellcycle.png"))
                logging.info("Plotting markers")
                cg.plot_markers(ds, out_file=os.path.join(out_dir, "L1_" + self.card + "_markers.png"))
                logging.info("Plotting neighborhood diagnostics")
                cg.plot_radius_characteristics(ds, out_file=os.path.join(out_dir, "L1_" + self.card + "_neighborhoods.png"))
                logging.info("Plotting batch covariates")
                cg.plot_batch_covariates(ds, out_file=os.path.join(out_dir, "L1_" + self.card + "_batches.png"))
                cg.ClusterValidator().fit(ds, os.path.join(out_dir, f"L1_{self.card}_cluster_pp.png"))
                logging.info("Plotting embedded velocity")
                cg.plot_embedded_velocity(ds, out_file=os.path.join(out_dir, f"L1_{self.card}_velocity.png"))
                logging.info("Plotting TFs")
                cg.plot_TFs(ds, dsagg, out_file_root=os.path.join(out_dir, f"L1_{self.card}"))
            except:
                pass
