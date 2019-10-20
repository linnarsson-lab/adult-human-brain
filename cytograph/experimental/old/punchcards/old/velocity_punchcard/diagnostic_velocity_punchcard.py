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


class DiagnosticVelocityPunchcard(luigi.Task):
    """Luigi Task to plot some diagnostic plots on the spliced, unspliced, ambigous molecules
    """
    card = luigi.Parameter()

    def requires(self) -> List[luigi.Task]:
        """
        Arguments
        ---------
        `ClusterL1`:
            passing ``card``
        `AggregateL1`:
            passing ``card``
        `ExportL1`:
            passing ``card``
        """
        # NOTE: not sure it needs Aggregate
        return [dm.ClusterPunchcard(card=self.card),
                dm.AggregatePunchcard(card=self.card),
                dm.ExportPunchcard(card=self.card)]

    def output(self) -> luigi.Target:
        """
        Returns
        -------
        file: ``L1_[card]_exported"/velocity_{self.card}_SAU_fractions.pdf``
        """
        return luigi.LocalTarget(os.path.join(dm.paths().build, f"{self.card}_exported", f"velocity_{self.card}_SAU_fractions.pdf"))

    def run(self) -> None:
        """Reads the output of `AggregateL1` and plots:
               - The fraction of splice, ambiguous, unspliced as ``velocity_[card]_exported/velocity_[card]_SAU_fractions.pdf``
        """
        logging = cg.logging(self, True)
        with self.output().temporary_path() as out_file:
            logging.info(f"Loading loom file in memory as a VelocytoLoom object for {self.card}")
            vlm = vcy.VelocytoLoom(self.input()[0].fn)
            logging.info(f"Plotting report on spliced, ambiguous, unpliced fraction for {self.card}")
            vlm.plot_fractions(save2file=out_file)
            # NOTE: other diagnostic plots if desired, but substitutet hte file for a path
