from typing import *
import os
import csv
import logging
import pickle
import loompy
import matplotlib.pyplot as plt
import numpy as np
import cytograph as cg
import luigi
from luigi import targets_map, EP2int, time_check, analysis_type_dict
from collections import defaultdict


def parse_project_requirements(process_obj: Dict) -> Iterator[luigi.Task]:
	parent_type = process_obj["type"]
	parent_kwargs = process_obj["kwargs"]
	if parent_type not in analysis_type_dict:
		raise NotImplementedError("type: %s not allowed, you need to allow it adding it to analysis_type_dict" % parent_type)
	Analysis = analysis_type_dict[parent_type]
	return Analysis(**parent_kwargs).requires()


class StudyProcess(luigi.WrapperTask):
	"""
	Luigi Task Wrapper to run a set of analysese on a particular slice of the data as specified by a description file

	`processname` needs to match th name specified in the .yaml file in the folder ../dev-processes
	"""
	
	processname = luigi.Parameter()

	def requires(self) -> Iterator[luigi.Task]:
		# TODO: Read the process object to know what to do instead of hard coding it
		# using
		# process_obj = cg.ProcessesParser()[self.processname]
		yield cg.StudyProcessPool(processname=self.processname)
		yield cg.ClusterLayoutProcess(processname=self.processname)
		yield cg.PlotGraphProcess(processname=self.processname)
