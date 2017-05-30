from typing import *
import os
import logging
from subprocess import check_output, CalledProcessError, Popen
import luigi
import cytograph as cg


class CellrangerCount(luigi.Task):
	"""
	A Luigi Task that runs "cellranger count" for a sample
	"""
	sample = luigi.Parameter()
	transcriptome = luigi.Parameter(default=os.path.join(cg.paths().transcriptome, "refdata-cellranger-mm10-1.2.0"))

	def requires(self) -> luigi.Task:
		return cg.CellrangerMkfastq(flowcell=self.flowcell)

	def output(self) -> luigi.LocalTarget:
		return luigi.LocalTarget(os.path.join(cg.paths.samples(), self.flowcell + "_" + self.sample + ".done"))
	
	def run(self) -> None:
		run = None  # type: str
		for f in os.listdir(cg.paths().runs):
			if f.__contains__(self.flowcell):
				run = f
		if run is None:
			raise FileNotFoundError("Run folder for " + self.flowcell + " not found.")
		
		cellranger_p = Popen(("cellranger", "count", "--id=" + self.sample, "--transcriptome=" + self.transcriptome, "--fastqs=" + self.flowcell + "/outs/fastq_path", "--sample=" + self.sample, "--expect-cells=3500"))
		cellranger_p.wait()
		if cellranger_p.returncode != 0:
			logging.error("cellranger exited with an error code")
			raise RuntimeError()
		else:
			with open(self.output().fn, "w") as f:
				f.write("This is just a placeholder")
