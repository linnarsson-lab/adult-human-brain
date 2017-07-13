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


class PerformAnalysis(luigi.WrapperTask):  # Status: check what it should return
	"""
	Luigi Task Wrapper to run a set of analyses on a particular slice of the data as specified by a description file

	`analysis` needs to match th name specified in the .yaml file in the folder ../cg-analysis
	"""
	
	analysis = luigi.Parameter()

	def requires(self) -> List[List[luigi.Task]]:
		analysis_obj = cg.AnalysesParser()[self.analysis]
		other_tasks = []
		for task in cg.parse_analysis_todo(analysis_obj):
			other_tasks.append(task(analysis=self.analysis))
		return [cg.ExportAnalysis(analysis=self.analysis), *other_tasks]  # Not sure why but before t was [[]]
