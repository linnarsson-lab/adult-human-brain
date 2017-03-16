"""
This module inspired by Spotify's Luigi SGE module, but for MPI
"""

import os
import sys
import pickle
import logging
from typing import *
import random
import subprocess
import luigi
import tempfile


class mpi(luigi.Config):
	hosts = luigi.Parameter(default="", description="Comma-separated list of hosts (empty: run locally)")
	shared_tmp_dir = luigi.Parameter(default='/data/tmp', significant=False)


class MpiTask(luigi.Task):
	def __init__(self, *args: str, **kwargs: Dict) -> None:
		super(MpiTask, self).__init__(*args, **kwargs)
		self.job_name = self.task_family

	def run(self) -> None:
		if mpi().hosts == "":
			self.work()
		else:
			self._run_job()

	def work(self) -> None:
		"""Override this method, rather than ``run()``,  for your actual work."""
		pass

	def _dump(self, out_dir: str) -> None:
		"""Dump instance to file."""
		with self.no_unpicklable_properties():
			self.job_file = os.path.join(out_dir, 'job-instance.pickle')
			pickle.dump(self, open(self.job_file, "w"))

	def _run_job(self) -> None:
		with tempfile.TemporaryDirectory(prefix=self.shared_tmp_dir) as tmp_dir:
			logging.debug("Dumping pickled class")
			self._dump(tmp_dir)

			runner_path = __file__
			if runner_path.endswith("pyc"):
				runner_path = runner_path[:-3] + "py"
			job_str = 'python {0} "{1}" "{2}"'.format(runner_path, self.tmp_dir, os.getcwd())  # enclose tmp_dir in quotes to protect from special escape chars

			# Build MPI submit command
			self.outfile = os.path.join(self.tmp_dir, 'job.out')
			self.errfile = os.path.join(self.tmp_dir, 'job.err')
			host = random.choice(self.hosts.split(","))
			submit_cmd = "mpirun --host " + host + " " + job_str + " > " + self.outfile + " 2> " + self.errfile
			logging.debug('command: \n' + submit_cmd)

			# Submit the job and wait for it to finish
			output = subprocess.check_output(submit_cmd, shell=True)


if __name__ == '__main__':
	work_dir = sys.argv[1]
	assert os.path.exists(work_dir), "First argument to mpi_runner.py must be a directory that exists"
	project_dir = sys.argv[2]
	sys.path.append(project_dir)
	os.chdir(work_dir)
	with open("job-instance.pickle", "r") as f:
		job = pickle.load(f)
	job.work()
