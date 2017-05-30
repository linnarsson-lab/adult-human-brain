from typing import *
import os
import logging
from subprocess import check_output, CalledProcessError, Popen
import luigi
import cytograph as cg


class CellrangerMkfastq(luigi.Task):
	"""
	A Luigi Task that runs "cellranger mkfastq" for a flowcell
	"""
	flowcell = luigi.Parameter()

	def requires(self) -> luigi.Task:
		return cg.FlowcellDefinition(flowcell=self.flowcell)

	def output(self) -> luigi.LocalTarget:
		return luigi.LocalTarget(os.path.join(cg.paths.samples(), self.flowcell + ".done"))
	
	def run(self) -> None:
		run = None  # type: str
		for f in os.listdir(cg.paths().runs):
			if f.__contains__(self.flowcell):
				run = f
		if run is None:
			raise FileNotFoundError("Run folder for " + self.flowcell + " not found.")
		
		cellranger_p = Popen(("cellranger", "mkfastq", "--run=" + os.path.join(cg.paths().runs, run), "--csv=" + self.flowcell + ".csv"))
		cellranger_p.wait()
		if cellranger_p.returncode != 0:
			logging.error("cellranger exited with an error code")
			raise RuntimeError()
		else:
			with open(self.output().fn, "w") as f:
				f.write("This is just a placeholder")
