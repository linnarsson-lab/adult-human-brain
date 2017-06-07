from typing import *
import os
import logging
import numpy as np
import pandas as pd
import re
import loompy


class CellTag:
	def __init__(self, category: str, file: str) -> None:
		self.category = category
		with open(file, "r") as f:
			for line in f:
				if line.startswith("name:"):
					self.name = line[5:].strip()
				if line.startswith("abbreviation:"):
					self.abbreviation = line[14:].strip()
				if line.startswith("definition:"):
					genes = line[12:].strip().split()
					self.positives = [x[1:] for x in genes if x.startswith("+")]
					self.negatives = [x[1:] for x in genes if x.startswith("-")]
				if line.startswith("categories:"):
					str_categories = line[11:].strip()
					self.categories = re.split(r"\W+",str_categories)
		if not hasattr(self, "name"):
			raise ValueError("'name' was missing")
		if not hasattr(self, "abbreviation"):
			raise ValueError("'abbreviation' was missing")
		if not hasattr(self, "positives"):
			raise ValueError("positive markers were missing")
		if not hasattr(self, "categories"):
			raise ValueError("categories were missing")
		if not hasattr(self, "negatives"):
			self.negatives = []

	def __str__(self) -> str:
		temp = self.name + " (" + self.abbreviation + "; " + " ".join(["+" + x for x in self.positives])
		if len(self.negatives) > 0:
			temp = temp + " " + " ".join(["-" + x for x in self.negatives]) + ")"
		else:
			temp = temp + ")"
		return temp


class AutoAnnotator(object):
	def __init__(self, root: str = "../auto-annotation") -> None:
		self.root = root
		self.tags = []  # type: List[CellTag]
		self.genes = None  # type: List[str]
		self.annotations = None  # type: np.ndarray
	
	def _load_defs(self) -> None:
		errors = False
		root_len = len(self.root)
		for cur, dirs, files in os.walk(self.root):
			for file in files:
				if file[-3:] == ".md" and file[-9:] != "README.md":
					try:
						tag = CellTag(cur[root_len:], os.path.join(cur, file))
						for pos in tag.positives:
							if pos not in self.genes:
								logging.error(file + ": gene '%s' not found in file", pos)
								errors = True
						for neg in tag.negatives:
							if neg not in self.genes:
								logging.error(file + ": gene '%s' not found in file", neg)
								errors = True
						self.tags.append(tag)
					except ValueError as e:
						logging.error(file + ": " + str(e))
						errors = True
		if errors:
			raise ValueError("Error loading cell tag definitions")
	
	@classmethod
	def load_direct(cls, root: str = "../auto-annotation") -> Any:
		"""
		Class method that loads the autoannotator from a folder without checking for genes
		In this way it can be used without the need of specifing a .loom file
		(e.g. self.genes can stay None and does not need to be filled in as is instead required by _load_defs)

		Args
		----
		root: str
			The directory containing the autoannotation files

		Returns
		-------
		aa: AutoAnnotator
			The autoannotator is loaded without checking for the existence of the genes.
			so self.genes = None

		Note
		----
		Usage is:
		aa = AutoAnnotator.load_direct()  # called on the class not an instance of the class
		"""
		errors = False
		aa = cls(root)
		root_len = len(aa.root)
		for cur, dirs, files in os.walk(aa.root):
			for file in files:
				if file[-3:] == ".md" and file[-9:] != "README.md":
					try:
						tag = CellTag(cur[root_len:], os.path.join(cur, file))
						aa.tags.append(tag)
					except ValueError as e:
						logging.error(file + ": " + str(e))
						errors = True
		if errors:
			raise ValueError("Error loading cell tag definitions")
		else:
			return aa

	def annotate_loom(self, ds: loompy.LoomConnection) -> np.ndarray:
		"""
		Annotate an already aggregated and trinarized loom file

		The input file should have one column per cluster and a layer named "trinaries"
		"""
		self.genes = ds.row_attrs["Gene"]
		trinaries = ds.layer["trinaries"]
		return self._do_annotate(trinaries)

	def annotate(self, in_file: str) -> np.ndarray:
		d = pd.read_csv(in_file, sep='\t', index_col=0)
		self.genes = d.index.values
		trinaries = d.values[:, :-1]
		return self._do_annotate(trinaries)

	def _do_annotate(self, trinaries: np.ndarray) -> np.ndarray:
		self._load_defs()

		self.annotations = np.empty((len(self.tags), trinaries.shape[1]))
		for ix, tag in enumerate(self.tags):
			for cluster in range(trinaries.shape[1]):
				p = 1
				for pos in tag.positives:
					index = np.where(self.genes == pos)[0][0]
					p = p * trinaries[index, cluster]
				for neg in tag.negatives:
					index = np.where(self.genes == neg)[0][0]
					p = p * (1 - trinaries[index, cluster])
				self.annotations[ix, cluster] = p
		return self.annotations
	
	def save(self, fname: str) -> None:
		with open(fname, "w") as f:
			f.write("Cluster\tTags\n")
			for ix in range(self.annotations.shape[1]):
				f.write(str(ix) + "\t")
				tags = []  # type: List[str]
				for j in range(self.annotations.shape[0]):
					if self.annotations[j, ix] > 0.5:
						tags.append(self.tags[j].abbreviation)
				tags.sort()
				f.write(",".join(tags))
				f.write("\n")

	def save_in_loom(self, ds: loompy.LoomConnection) -> None:
		attr = []
		for ix in range(self.annotations.shape[1]):
			tags = []  # type: List[str]
			for j in range(self.annotations.shape[0]):
				if self.annotations[j, ix] > 0.5:
					tags.append(self.tags[j].abbreviation)
			tags.sort()
			attr.append(",".join(tags))
		ds.set_attr("AutoAnnotation", np.array(attr), axis=1)


def read_autoannotation(aa_file: str) -> List[List[str]]:
	"""Extract autoannotations from file

	Arguments

	Returns
	-------
	tags : List[List[str]]
		where tags[i] contains all the aa tags attributed to cluster i
	"""
	tags = []  # type: list
	with open(aa_file, "r") as f:
		content = f.readlines()[1:]
		for line in content:
			tags.append(line.rstrip("\n").split('\t')[1].split(","))
	return tags
