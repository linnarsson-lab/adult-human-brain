from typing import *
import os
import logging
import luigi
import cytograph as cg


class FlowcellDefinition(luigi.ExternalTask):
    """
    A Luigi Task that simply returns the existing .csv file for a flowcell
    """
    flowcell = luigi.Parameter()

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(os.path.join(cg.paths.samples(), self.flowcell + ".csv"))
