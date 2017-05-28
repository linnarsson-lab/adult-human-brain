from typing import *
import luigi


class paths():
    samples = luigi.Parameter(default="/data/proj/chromium/loom")
    build = luigi.Parameter()

