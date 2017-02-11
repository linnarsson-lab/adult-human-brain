from typing import *
import csv


class Sample:
	def __init__(self, sample: str, tissue: str, project: str) -> None:
		self.sample = sample
		self.tissue = tissue
		self.project = project


class PoolSpec:
	def __init__(self, fname: str = "pooling_specification.tab") -> None:
		self.samples = []  # type: List[Sample]
	
		with open(fname, 'r') as f:
			for row in csv.reader(f, delimiter="\t"):
				if row[2] != "FAILED":
					self.samples.append(Sample(row[0], row[1], row[3]))
	
	@property
	def tissues(self) -> List[str]:
		return list(set([s.tissue for s in self.samples]))
	
	@property
	def projects(self) -> List[str]:
		return list(set([s.project for s in self.samples]))

	def samples_for_tissue(self, tissue: str) -> List[str]:
		return list(set([s.sample for s in self.samples if s.tissue == tissue]))
	
	def tissues_for_project(self, project: str) -> List[str]:
		return list(set([s.tissue for s in self.samples if s.project == project]))

