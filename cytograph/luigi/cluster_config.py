from typing import *
import luigi


class cluster(luigi.Config):
    method = luigi.Parameter(default="dbscan")  # 'dbscan', hdbscan', 'lj'