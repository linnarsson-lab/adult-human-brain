from typing import *
import os
import loompy
from scipy import sparse
import numpy as np
import cytograph as cg
import development_mouse as dm
import velocyto as vcy
import matplotlib.pyplot as plt
import luigi


class EstimateVelocityPunchcard(luigi.Task):
    """Luigi Task to run velocyto
    """
    card = luigi.Parameter()

    def requires(self) -> List[luigi.Task]:
        """
        Arguments
        ---------
        `ClusterPunchcard`:
            passing ``card``
        `AggregatePunchcard`:
            passing ``card``
        """
        return [dm.ClusterPunchcard(card=self.card),
                dm.AggregatePunchcard(card=self.card)]

    def output(self) -> luigi.Target:
        """
        Returns
        -------
        file: ``velocity_[CARD].hdf5``
        """
        return luigi.LocalTarget(os.path.join(dm.paths().build, f"velocity_{self.card}.hdf5"))

    def run(self) -> None:
        """Run the velocity inference (without the projection on tsne) and generate the file:
            velocity_[CARD].hdf5
        """
        logging = cg.logging(self, True)
        with self.output().temporary_path() as out_file:
            
            logging.info("Loading loom file in memory as a VelocytoLoom object")
            vlm = vcy.VelocytoLoom(self.input()[0].fn)

            logging.info("Use `Cluster` column_attr to set clusters and _X, _Y to set the tsne embedding")
            vlm.set_clusters(cluster_labels=vlm.ca["Clusters"])
            vlm.ts = np.column_stack([vlm.ca["_X"], vlm.ca["_Y"]])  # load the embedding from previous analysis

            # NOTE: code below is basically identical to `default_filter_and_norm` but with the exception of adjust_totS_totU
            # Heuristics, we should set better heuristic and could add a config file with parameters for analysis
            max_expr_avg = 40
            min_expr_counts = max(20, min(100, vlm.S.shape[1] * 2.25e-3))
            min_cells_express = max(10, min(50, vlm.S.shape[1] * 1.5e-3))
            N = max(1000, min(int((vlm.S.shape[1] / 1000)**(1 / 3) / 0.0008), 5000))
            min_avg_U = 0.01
            min_avg_S = 0.08

            # NOTE: not sure if this is needed with the new init
            vlm.normalize("S", size=True, log=False)
            vlm.normalize("U", size=True, log=False)

            logging.info("Performing gene filtering by S detection")
            vlm.score_detection_levels(min_expr_counts=min_expr_counts, min_cells_express=min_cells_express)
            vlm.filter_genes(by_detection_levels=True)

            logging.info("Performing gene filtering by Cv vs mean relation")
            vlm.score_cv_vs_mean(N=N, max_expr_avg=max_expr_avg)
            vlm.filter_genes(by_cv_vs_mean=True)

            logging.info("Performing gene filtering by U detection")
            vlm.score_detection_levels(min_expr_counts=0, min_cells_express=0,
                                       min_expr_counts_U=int(min_expr_counts / 2) + 1,
                                       min_cells_express_U=int(min_cells_express / 2) + 1)
            
            if hasattr(vlm, "cluster_labels"):
                logging.info("Performing gene filtering by cluster expression")
                vlm.score_cluster_expression(min_avg_U=min_avg_U, min_avg_S=min_avg_S)
                vlm.filter_genes(by_detection_levels=True, by_cluster_expression=True)
            else:
                vlm.filter_genes(by_detection_levels=True)

            vlm.normalize_by_total(plot=True)

            logging.info("Preparing dataset for velocity extimation")
            vlm.perform_PCA()
            n_comps = int(np.where(np.diff(np.diff(np.cumsum(vlm.pca.explained_variance_ratio_)) > 0.002))[0][0])
            n_comps = min(n_comps, 50)
            k = int(min(1000, max(10, np.ceil(vlm.S.shape[1] * 0.02))))

            logging.info(f"Considering {n_comps} components and {k} nearest neighbours")
            vlm.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k * 8, b_maxl=k * 4, n_jobs=8)

            vlm.normalize_median()  # NOTE: it had problems in the past...

            logging.info(f"Fitting gammas for {vlm.Sx_sz.shape[1]} genes")
            vlm.fit_gammas()

            logging.info("Calculate velocity")
            vlm.predict_U()
            vlm.calculate_velocity()
            vlm.calculate_shift(assumption="constant_velocity")
            vlm.extrapolate_cell_at_t(delta_t=1)  # NOTE: we should determine delta t in a better way

            vlm.to_hdf5(out_file)
