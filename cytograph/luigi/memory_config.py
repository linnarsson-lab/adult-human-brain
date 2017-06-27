from typing import *
import luigi


class memory(luigi.Config):
    batch = luigi.IntParameter(default=1000)
    batchrows = luigi.IntParameter(default=0)
    batchcolumns = luigi.IntParameter(default=0)

    @property
    def axis0(self) -> Any:
        if self.batchrows == 0:
            return self.batch
        return self.batchrows

    @property
    def axis1(self) -> Any:
        if self.batchcolumns == 0:
            return self.batch
        return self.batchcolumns
