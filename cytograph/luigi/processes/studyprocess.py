from typing import *
import os
import csv
import logging
import pickle
import loompy
import numpy as np
import cytograph as cg
import luigi
from collections import defaultdict


class StudyProcess(luigi.WrapperTask):
	"""
	Luigi Task Wrapper to run a set of analyses on a particular slice of the data as specified by a description file

	`processname` needs to match th name specified in the .yaml file in the folder ../dev-processes
	"""
	
	processname = luigi.Parameter()

	def requires(self) -> List[List[luigi.Task]]:
		process_obj = cg.ProcessesParser()[self.processname]
		other_tasks = []
		for task in cg.parse_project_todo(process_obj):
			other_tasks.append(task(processname=self.processname))
		return [[cg.ClusterLayoutProcess(processname=self.processname), cg.AutoAnnotateProcess(processname=self.processname), *other_tasks]]
