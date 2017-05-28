"""
This module inspired by Spotify's Luigi SGE module, but using SSH to run on a simple private cluster
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
import re
import socket


class parallel(luigi.Config):
	hosts = luigi.Parameter(default="", description="Comma-separated list of hosts (empty: run locally)")
	shared_tmp_dir = luigi.Parameter(default='/data/tmp', significant=False)
	all_hosts = luigi.ListParameter(default=["monod01", "monod02", "monod03", "monod04", "monod05", "monod06", "monod07", "monod08", "monod09", "monod10", "monod11", "monod12"])


class ParallelTask(luigi.Task):
	def __init__(self, *args: str, **kwargs: Dict) -> None:
		super(ParallelTask, self).__init__(*args, **kwargs)
		self.job_name = self.task_family
		if parallel().hosts == "":
			self.hosts = []  # type: List[str]
		elif parallel().hosts == "all":
			self.hosts = parallel().all_hosts
		elif re.fullmatch("[0-1][0-9]-[0-1][0-9]", parallel().hosts):
			low = int(parallel().hosts.split("-")[0]) - 1
			hi = int(parallel().hosts.split("-")[0])
			self.hosts = parallel().all_hosts[low:hi]

	def run(self) -> None:
		if self.hosts == "":
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
		logging.info("Running task on " + socket.gethostname().split('.')[0])
		with tempfile.TemporaryDirectory(prefix=self.shared_tmp_dir) as tmp_dir:
			logging.debug("Dumping pickled class")
			self._dump(tmp_dir)

			runner_path = __file__
			if runner_path.endswith("pyc"):
				runner_path = runner_path[:-3] + "py"
			job_str = 'python {0} "{1}" "{2}"'.format(runner_path, self.tmp_dir, os.getcwd())  # enclose tmp_dir in quotes to protect from special escape chars

			# Build ssh command
			self.outfile = os.path.join(self.tmp_dir, 'job.out')
			self.errfile = os.path.join(self.tmp_dir, 'job.err')
			host = random.choice(self.hosts)
			submit_cmd = "ssh " + host + " " + job_str + " > " + self.outfile + " 2> " + self.errfile
			logging.debug('command: \n' + submit_cmd)

			# Submit the job and wait for it to finish
			output = subprocess.check_output(submit_cmd, shell=True)


class TestParallelTask(ParallelTask):
	def __init__(self, *args: str, **kwargs: Dict) -> None:
		# I'm getting a type error on this line, but I think it's a mypy bug
		super(TestParallelTask, self).__init__(*args, **kwargs)

	def requires(self) -> luigi.Task:
		return []

	def output(self) -> luigi.Target:
		return luigi.LocalTarget(os.path.join(cg.paths().build, "parallel_test.txt"))
	
	def work(self) -> None:
		logging.info("Running task on " + socket.gethostname().split('.')[0])
		with self.output().temporary_path() as out_file:
			with open(out_file, "w") as f:
				f.write("This file is just a placeholder\n")


if __name__ == '__main__':
	work_dir = sys.argv[1]
	assert os.path.exists(work_dir), "First argument must be a directory that exists"
	project_dir = sys.argv[2]
	sys.path.append(project_dir)
	os.chdir(work_dir)
	with open("job-instance.pickle", "r") as f:
		job = pickle.load(f)
	job.work()
