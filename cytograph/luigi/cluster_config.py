from typing import *
import luigi


class cluster(luigi.Config):
    method = luigi.Parameter(default="dbscan")  # 'dbscan', hdbscan', 'lj'
    no_outliers = luigi.BoolParameter(default=False)
