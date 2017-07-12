from typing import *
import luigi


class skip(luigi.Config):
    classifier = luigi.BoolParameter(default=False)
