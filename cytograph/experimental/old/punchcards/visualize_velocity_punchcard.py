from typing import *
import os
import loompy
from scipy import sparse
import numpy as np
import cytograph as cg
import development_mouse as dm
import velocyto as vcy
import matplotlib.pyplot as plt
import tempfile
import luigi


class VisualizeVelocityPunchcard(luigi.Task):
    """Luigi Task to run velocyto
    """
    card = luigi.Parameter(description="Name of the punchcard to run")

    def requires(self) -> List[luigi.Task]:
        """
        Arguments
        ---------
        `EstimateVelocity`:
            passing ``card``
        """
        # NOTE: not sure it needs AggregateL1
        return [dm.EstimateVelocityPunchcard(card=self.card), dm.AggregatePunchcard(card=self.card)]

    def output(self) -> luigi.Target:
        """
        Returns
        -------
        folder: ``velocity_[CARD]_export``:
            Note this is kind of a hack to luigi, single files will not be regenerated but whole folder will.
        """
        return luigi.LocalTarget(os.path.join(dm.paths().build, f"velocity_{self.card}_export"))

    def run(self) -> None:
        """Run ``estimate_transition_prob`` and output the result as a png
        """
        logging = cg.logging(self, True)
        logging.info("Exporting cluster data")
        with self.output().temporary_path() as out_dir:
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            logging.info(f"Loading tags from the exported file {self.input()[1].fn}")
            dsagg = loompy.connect(self.input()[1].fn)
            tags = list(dsagg.col_attrs["AutoAnnotation"])
            dsagg.close()

            logging.info(f"Loading {self.input()[0].fn} as a VelocytoLoom object")
            vlm = vcy.load_velocyto_hdf5(self.input()[0].fn)

            logging.info(f"Estimating transition probability. Note: This step will require a bit of time")
            n_neighbors = int(vlm.S.shape[1] / 5)
            vlm.estimate_transition_prob(hidim="Sx_sz", embed="ts", transform="sqrt",
                                         n_neighbors=n_neighbors, knn_random=True, sampled_fraction=1,
                                         n_jobs=dm.threads().limit, threads=dm.threads().limit)
            # NOTE here we might want to tune the number of jobs used in relation to the number of concurrent tasks
            # NOTE here we might want to change the sampled fraction to a lower number to make things faster

            logging.info("Serializing the vlm object. This might take long time and generate huge file on disk.")
            
            # NOTE: IMPORTANT here I actually modify a luigi Target, this is not considered good practice
            # Dump to a temp and only substitute the original file, atomically just before the folder velocity_[CARD]_export get created
            tmp_file = tempfile.mktemp(dir=os.path.join(dm.paths().build))
            # I get if I leave the default dir: OSError: [Errno 18] Invalid cross-device link: '/tmp/tmphnvw5x6_' -> '/data/proj/development/build_20171115/velocity_Forebrain_E9-11.hdf5'
            
            # Run the abstracted graph and velocity summary
            confidence = cg.adjacency_confidence(vlm.knn.tocoo(), vlm.cluster_ix, symmetric=True)
            significant, trans, expected_tr = cg.velocity_summary(vlm)
            vlm.dm_confidence, vlm.dm_significant, vlm.dm_trans, vlm.dm_expected_tr = confidence, significant, trans, expected_tr

            # NOTE: This was placed here for testing purposes, after testing put as last plot
            plt.figure(None, (14, 14))
            dm.plot_velocity_summary(vlm, confidence, significant, trans, expected_tr,
                                     out_file=os.path.join(out_dir, "velocity_" + self.card + "_summary.png"),
                                     tags=tags)

            vlm.to_hdf5(tmp_file)

            vlm.calculate_embedding_shift(sigma_corr=0.05)  # NOTE: this parameter could be tuned

            vlm.calculate_grid_arrows(smooth=0.8, steps=(40, 40), n_neighbors=300)  # NOTE: this parameters could be tuned

            plt.figure(None, (14, 14))
            # NOTE: the one below should be updated to include autoannotaiton and legend
            vlm.plot_grid_arrows(scatter_kwargs_dict={"alpha": 0.35, "lw": 0.35, "edgecolor": "0.4", "s": 38, "rasterized": True},
                                 min_mass=10, angles='xy', scale_units='xy',
                                 headaxislength=2.75, headlength=5, headwidth=4.8, quiver_scale=0.25)

            # NOTE: this parameters could be tuned. In particular min_mass!
            plt.savefig(os.path.join(out_dir, "velocity_" + self.card + "_TSNE.png"))

            plt.figure(None, (9, 9))
            cg.plot_confidence_and_velocity(trans, expected_tr, confidence)
            plt.savefig(os.path.join(out_dir, "velocity_" + self.card + "_transitions.png"))
            
            os.rename(tmp_file, os.path.join(dm.paths().build, f"velocity_{self.card}.hdf5"))  # Atomic substitution of a previous file
