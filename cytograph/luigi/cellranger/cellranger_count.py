from typing import *
import os
import logging
from subprocess import check_output, CalledProcessError, Popen
import luigi
import cytograph as cg


class CellrangerCount(luigi.Task):
	"""
	A Luigi Task that runs "cellranger count" for a flowcell
	"""
	flowcell = luigi.Parameter()
	transcriptome = luigi.Parameter(default=os.path.join(cg.paths().transcriptome, "refdata-cellranger-mm10-1.2.0"))
	nodes = luigi.Parameter(default="monod05,monod06,monod07,monod08,monod09,monod10,monod11,monod12")
	n_cells = luigi.Parameter(default="3500,3500,3500,3500,3500,3500,3500,3500")

	def requires(self) -> luigi.Task:
		return cg.CellrangerMkfastq(flowcell=self.flowcell)

	def output(self) -> luigi.LocalTarget:
		return luigi.LocalTarget(os.path.join(cg.paths.samples(), self.flowcell + ".done"))
	
	def run(self) -> None:
		run = None  # type: str
		for f in os.listdir(cg.paths().runs):
			if f.__contains__(self.flowcell):
				run = f
		if run is None:
			raise FileNotFoundError("Run folder for " + self.flowcell + " not found.")

		n_cells = self.n_cells.split(",")
		procs = []
		for ix, sample in enumerate(samples):
			proc = Popen(("ssh " + self.nodes[ix] + " cellranger", "count", "--id=" + sample, "--transcriptome=" + self.transcriptome, "--fastqs=" + self.flowcell + "/outs/fastq_path", "--sample=" + sample, "--expect-cells=" + n_cells[ix]))
			procs.append(proc)
		for ix, proc in enumerate(procs):
			proc.wait()
			if proc.returncode != 0:
				logging.error("cellranger error for sample: " + samples[ix])
				raise RuntimeError()
		with open(self.output().fn, "w") as f:
			f.write("This is just a placeholder")
