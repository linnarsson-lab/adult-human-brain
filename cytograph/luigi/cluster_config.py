from typing import *
import luigi


class clustering(luigi.Config):
    method = luigi.Parameter(default="dbscan")  # 'dbscan', hdbscan', 'lj'
