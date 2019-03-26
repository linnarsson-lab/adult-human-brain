
# Adapted from SGEJobTask by Alex Wiltschko (@alexbw)
# Original license below
# This version written by Sten Linnarsson 2019

#
# Copyright 2012-2015 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# This extension is modeled after the hadoop.py approach.
#
# Implementation notes
# The procedure:
# - Pickle the class
# - Construct a qsub argument that runs a generic runner function with the path to the pickled class
# - Runner function loads the class from pickle
# - Runner function hits the work button on it

import os
import subprocess
import time
import sys
import logging
import random
import pickle
import luigi
from luigi.contrib.hadoop import create_packages_archive
from luigi.contrib import sge_runner
import logging

POLL_TIME = 5  # decided to hard-code rather than configure here


def _parse_qstat_state(qstat_out, job_id):
	"""Parse "state" column from `qstat` output for given job_id

	Returns state for the *first* job matching job_id. Returns 'u' if
	`qstat` output is empty or job_id is not found.

	"""
	if qstat_out.strip() == '':
		return 'u'
	lines = qstat_out.split('\n')
	# skip past header
	while not lines.pop(0).startswith('---'):
		pass
	for line in lines:
		if line:
			job, prior, name, user, state = line.strip().split()[0:5]
			if int(job) == int(job_id):
				return state
	return 'u'


def _parse_qsub_job_id(qsub_out):
	"""Parse job id from qsub output string.

	Assume format:

		"Your job <job_id> ("<job_name>") has been submitted"

	"""
	return int(qsub_out.split()[2])


def _build_qsub_command(cmd, job_name, outfile, errfile, pe, n_cpu):
	"""Submit shell command to SGE queue via `qsub`"""
	qsub_template = """echo {cmd} | qsub -o ":{outfile}" -e ":{errfile}" -V -r y -pe {pe} {n_cpu} -N {job_name}"""
	return qsub_template.format(
		cmd=cmd, job_name=job_name, outfile=outfile, errfile=errfile,
		pe=pe, n_cpu=n_cpu)


class CondorTask(luigi.Task):

	"""Base class for executing a job on a HTCondor cluster

	Override ``work()`` (rather than ``run()``) with your job code.

	Parameters:

	- n_cpu: Number of CPUs (or "slots") to allocate for the Task. This
		value is passed as ``qsub -pe {pe} {n_cpu}``
	- parallel_env: SGE parallel environment name. The default is "orte",
		the parallel environment installed with MIT StarCluster. If you
		are using a different cluster environment, check with your
		sysadmin for the right pe to use. This value is passed as {pe}
		to the qsub command above.
	- shared_tmp_dir: Shared drive accessible from all nodes in the cluster.
		Task classes and dependencies are pickled to a temporary folder on
		this drive. The default is ``/home``, the NFS share location setup
		by StarCluster
	- job_name_format: String that can be passed in to customize the job name
		string passed to qsub; e.g. "Task123_{task_family}_{n_cpu}...".
	- job_name: Exact job name to pass to qsub.
	- run_locally: Run locally instead of on the cluster.
	- poll_time: the length of time to wait in order to poll qstat
	- dont_remove_tmp_dir: Instead of deleting the temporary directory, keep it.
	- no_tarball: Don't create a tarball of the luigi project directory.  Can be
		useful to reduce I/O requirements when the luigi directory is accessible
		from cluster nodes already.

	"""

	request_cpus = luigi.IntParameter(default=2, significant=False)
	request_gpus = luigi.IntParameter(default=0, significant=False)

	shared_tmp_dir = luigi.Parameter(default='/home', significant=False)
	poll_time = luigi.IntParameter(
		significant=False, default=POLL_TIME,
		description="specify the wait time to poll qstat for the job status")

	def __init__(self, *args, **kwargs):
		super(CondorTask, self).__init__(*args, **kwargs)

	def _fetch_task_failures(self):
		if not os.path.exists(self.errfile):
			logger.info('No error file')
			return []
		with open(self.errfile, "r") as f:
			errors = f.readlines()
		if errors == []:
			return errors
		if errors[0].strip() == 'stdin: is not a tty':  # SGE complains when we submit through a pipe
			errors.pop(0)
		return errors

	def run(self):
		if self.run_locally:
			self.work()
		else:
			self._dump(self.tmp_dir)
			self._run_job()

			# The procedure:
			# - Pickle the class
			# - Tarball the dependencies
			# - Construct a qsub argument that runs a generic runner function with the path to the pickled class
			# - Runner function loads the class from pickle
			# - Runner class untars the dependencies
			# - Runner function hits the button on the class's work() method

	def work(self):
		"""Override this method, rather than ``run()``,  for your actual work."""
		pass

	def _dump(self, out_dir=''):
		"""Dump instance to file."""
		with self.no_unpicklable_properties():
			self.job_file = os.path.join(out_dir, 'job-instance.pickle')
			if self.__module__ == '__main__':
				d = pickle.dumps(self)
				module_name = os.path.basename(sys.argv[0]).rsplit('.', 1)[0]
				d = d.replace('(c__main__', "(c" + module_name)
				with open(self.job_file, "w") as f:
					f.write(d)
			else:
				with open(self.job_file, "w") as f:
					pickle.dump(self, f)

	def _run_job(self):

		# Build a qsub argument that will run sge_runner.py on the directory we've specified
		runner_path = sge_runner.__file__
		if runner_path.endswith("pyc"):
			runner_path = runner_path[:-3] + "py"
		job_str = 'python {0} "{1}" "{2}"'.format(
			runner_path, self.tmp_dir, os.getcwd())  # enclose tmp_dir in quotes to protect from special escape chars
		if self.no_tarball:
			job_str += ' "--no-tarball"'

		# Build qsub submit command
		self.outfile = os.path.join(self.tmp_dir, 'job.out')
		self.errfile = os.path.join(self.tmp_dir, 'job.err')
		submit_cmd = _build_qsub_command(job_str, self.task_family, self.outfile,
										 self.errfile, self.parallel_env, self.n_cpu)
		logger.debug('qsub command: \n' + submit_cmd)

		# Submit the job and grab job ID
		output = subprocess.check_output(submit_cmd, shell=True)
		self.job_id = _parse_qsub_job_id(output)
		logger.debug("Submitted job to qsub with response:\n" + output)

		self._track_job()

		# Now delete the temporaries, if they're there.
		if (self.tmp_dir and os.path.exists(self.tmp_dir) and not self.dont_remove_tmp_dir):
			logger.info('Removing temporary directory %s' % self.tmp_dir)
			subprocess.call(["rm", "-rf", self.tmp_dir])

	def _track_job(self):
		while True:
			# Sleep for a little bit
			time.sleep(self.poll_time)

			# See what the job's up to
			# ASSUMPTION
			qstat_out = subprocess.check_output(['qstat'])
			sge_status = _parse_qstat_state(qstat_out, self.job_id)
			if sge_status == 'r':
				logger.info('Job is running...')
			elif sge_status == 'qw':
				logger.info('Job is pending...')
			elif 'E' in sge_status:
				logger.error('Job has FAILED:\n' + '\n'.join(self._fetch_task_failures()))
				break
			elif sge_status == 't' or sge_status == 'u':
				# Then the job could either be failed or done.
				errors = self._fetch_task_failures()
				if not errors:
					logger.info('Job is done')
				else:
					logger.error('Job has FAILED:\n' + '\n'.join(errors))
				break
			else:
				logger.info('Job status is UNKNOWN!')
				logger.info('Status is : %s' % sge_status)
				raise Exception("job status isn't one of ['r', 'qw', 'E*', 't', 'u']: %s" % sge_status)

