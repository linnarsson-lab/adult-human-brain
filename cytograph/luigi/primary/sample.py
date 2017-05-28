from typing import *
import os
import logging
import luigi
import cytograph as cg

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)


class Sample(luigi.ExternalTask):
    """
    A Luigi Task that simply returns the existing raw Loom file for a sample

    TODO: check if the file exists and if not, download from Google Cloud Storage
    """
    sample = luigi.Parameter()

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(os.path.join(cg.paths.samples(), self.sample + ".loom"))
