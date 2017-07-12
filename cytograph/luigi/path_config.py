from typing import *
import luigi


class paths(luigi.Config):
    samples = luigi.Parameter(default="/data/proj/chromium/loom")
    build = luigi.Parameter()
    runs = luigi.Parameter(default="/data/runs")
    transcriptome = luigi.Parameter(default="/data/ref/cellranger/")
    use_velocyto = luigi.BoolParameter(default=False)
    autoannotation = luigi.Parameter(default="../auto-annotation")
