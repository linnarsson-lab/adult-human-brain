from typing import *
import os
import logging
import luigi
import cytograph as cg


class Sample(luigi.ExternalTask):
    """
    A Luigi Task that simply returns the existing raw Loom file for a sample

    TODO: check if the file exists and if not, download from Google Cloud Storage
    """
    sample = luigi.Parameter()

    def output(self) -> luigi.LocalTarget:
        if cg.paths().use_velocyto:
            fname = os.path.join(cg.paths().samples, self.sample, "velocyto", self.sample + ".loom")
            return luigi.LocalTarget(fname)
        else:
            fname = os.path.join(cg.paths().samples, self.sample, self.sample + ".loom")
            if os.path.exists(fname):
                return luigi.LocalTarget(fname)
            else:
                fname = os.path.join(cg.paths().samples, self.sample + ".loom")
                return luigi.LocalTarget(fname)
