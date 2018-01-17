from typing import *
import os
import logging
import numpy as np
import pandas as pd
import re
import loompy


class CellTag:
	unknown_tags: set = set()

	def __init__(self, category: str, file: str) -> None:
		self.category: str = category
		self.name: str = ""
		self.description: str = ""
		self.abbreviation: str = None
		at_descr = False
		in_synonyms = False
		with open(file, "r", encoding="utf-8") as f:
			for line in f:
				if line.startswith("---"):
					at_descr = True
					continue
				if at_descr:
					self.description += line
					continue
				if line.startswith("synonyms:"):
					self.synonyms: list = []
					in_synonyms = True
					continue
				if line.startswith("- ") and in_synonyms:
					self.synonyms.append(line[2:].strip())
					continue
				in_synonyms = False
				m = re.search("version: *([0-9])+", line)
				if m:
					self.version = m.group(1)
					line = line[ :m.start()]
				# if line.startswith("name:"):
				# 	self.name = line[5:].strip()
				# if line.startswith("abbreviation:"):
				# 	self.abbreviation = line[14:].strip()
				if line.startswith("definition:"):
					genes = line[12:].strip().split()
					self.positives = [x[1:] for x in genes if x.startswith("+")]
					self.negatives = [x[1:] for x in genes if x.startswith("-")]
					continue
				if line.startswith("categories:"):
					str_categories = line[11:].strip()
					self.categories = re.split(r"\W+",str_categories)
					continue
				if ":" in line:
					tagid, value = line.strip().split(":", 1)
					self.__setattr__(tagid.strip(), value.strip())
				elif len(line.strip()) > 0:
					print(self.name + " Unknown data in " + line)
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
		self.description = self.description.strip()

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
	
	def _load_defs(self, from_yaml: bool = False) -> None:
		fileext = ".yaml" if from_yaml else ".md"
		if from_yaml:
			import yaml
		errors = False
		root_len = len(self.root)
		for cur, dirs, files in os.walk(self.root):
			for file in files:
				if file.endswith(fileext) and file[-9:] != "README.md":
					try:
						if from_yaml:
							tag = yaml.load(open(os.path.join(cur, file)))
						else:
							tag = CellTag(cur[root_len:], os.path.join(cur, file))
						for pos in tag.positives:
							if (self.genes is not None) and (pos not in self.genes):
								logging.error(file + ": gene '%s' not found in file", pos)
								errors = True
						for neg in tag.negatives:
							if (self.genes is not None) and (neg not in self.genes):
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
						print(file)
						tag = CellTag(cur[root_len:], os.path.join(cur, file))
						aa.tags.append(tag)
					except ValueError as e:
						logging.error(file + ": " + str(e))
						errors = True
		if errors:
			raise ValueError("Error loading cell tag definitions")
		else:
			return aa

	def yaml_dump_annotations(self, root_path: str) -> None:
		import yaml
		for tag in self.tags:
			try:
				sub_path = tag.category
				while sub_path.startswith("/"):
					sub_path = sub_path[1:]
				tag_path = os.path.join(root_path, sub_path)
			except:
				tag_path = root_path
			tag_yaml = yaml.dump(tag)
			os.makedirs(tag_path, exist_ok=True)
			with open(os.path.join(tag_path, tag.abbreviation + ".yaml"), 'w') as f:
				f.write(tag_yaml)

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
		ds.ca.AutoAnnotation = np.array(attr)

def read_autoannotation(aa_file: str) -> List[List[str]]:
	"""DEPRECATED
	Extract autoannotations from file

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
