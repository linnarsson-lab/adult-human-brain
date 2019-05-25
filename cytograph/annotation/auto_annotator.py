import logging
import os
import re
from typing import List

import numpy as np
import yaml

import loompy


class Annotation:
	unknown_tags: set = set()

	def __init__(self, category: str, filename: str) -> None:
		with open(filename) as f:
			doc = next(yaml.load_all(f))

		if "name" in doc:
			self.name = doc["name"]
		else:
			raise ValueError(os.path.basename(filename) + " did not contain a 'name' attribute, which is required.")

		if "abbreviation" in doc:
			self.abbreviation = doc["abbreviation"]
		else:
			raise ValueError(os.path.basename(filename) + " did not contain an 'abbreviation' attribute, which is required.")

		if "definition" in doc:
			self.definition = doc["definition"]
			genes = self.definition.strip().split()
			self.positives = [x[1:] for x in genes if x.startswith("+")]
			self.negatives = [x[1:] for x in genes if x.startswith("-")]
		else:
			raise ValueError(os.path.basename(filename) + " did not contain a 'definition' attribute, which is required.")

		if "categories" in doc and doc["categories"] is not None:
			self.categories = re.split(r"\W+", doc["categories"].strip())
		else:
			self.categories = []

	def __str__(self) -> str:
		temp = self.name + " (" + self.abbreviation + "; " + " ".join(["+" + x for x in self.positives])
		if len(self.negatives) > 0:
			temp = temp + " " + " ".join(["-" + x for x in self.negatives]) + ")"
		else:
			temp = temp + ")"
		return temp


class AutoAnnotator(object):
	def __init__(self, root: str, ds: loompy.LoomConnection = None) -> None:
		self.root = root
		self.definitions: List[Annotation] = []
		self.genes: List[str] = [] if ds is None else ds.ra.Gene
		self.annotations = None  # type: np.ndarray
	
		fileext = [".yaml", ".md"]
		root_len = len(self.root)
		for cur, _, files in os.walk(self.root):
			for file in files:
				errors = False
				if os.path.splitext(file)[-1] in fileext and file[-9:] != "README.md":
					try:
						tag = Annotation(cur[root_len:], os.path.join(cur, file))
						for pos in tag.positives:
							if len(self.genes) > 0 and (pos not in self.genes):
								logging.error(file + ": gene '%s' not found in file", pos)
								errors = True
						for neg in tag.negatives:
							if len(self.genes) > 0 and (neg not in self.genes):
								logging.error(file + ": gene '%s' not found in file", neg)
								errors = True
						if not errors:
							self.definitions.append(tag)
					except Exception as e:
						logging.error(file + ": " + str(e))
						errors = True
		# if errors:
		# 	raise ValueError("Error loading cell tag definitions")
	
	def fit(self, ds: loompy.LoomConnection) -> np.ndarray:
		"""
		Return the annotation for an already aggregated and trinarized loom file

		The input file should have one column per cluster and a layer named "trinaries"

		Returns:
			An array of strings giving the auto-annotation for each cluster
		"""
		self.genes = ds.ra.Gene
		trinaries = ds.layers["trinaries"]
		self.annotations = np.empty((len(self.definitions), trinaries.shape[1]))
		for ix, tag in enumerate(self.definitions):
			for cluster in range(trinaries.shape[1]):
				p = 1
				for pos in tag.positives:
					if pos not in self.genes:
						logging.error(f"Auto-annotation gene {pos} (used for {tag}) not found in file")
						continue
					index = np.where(self.genes == pos)[0][0]
					p = p * trinaries[index, cluster]
				for neg in tag.negatives:
					if neg not in self.genes:
						logging.error(f"Auto-annotation gene {neg} (used for {tag}) not found in file")
						continue
					index = np.where(self.genes == neg)[0][0]
					p = p * (1 - trinaries[index, cluster])
				self.annotations[ix, cluster] = p

		attr = []
		for ix in range(self.annotations.shape[1]):
			tags = []  # type: List[str]
			for j in range(self.annotations.shape[0]):
				if self.annotations[j, ix] > 0.5:
					tags.append(self.definitions[j].abbreviation)
			tags.sort()
			attr.append(" ".join(tags))
		
		return np.array(attr)

	def annotate(self, ds: loompy.LoomConnection) -> None:
		"""
		Annotate an aggregated and trinarized loom file


		Remarks:
			Creates the following new column attributes:
				AutoAnnotation:		Space-separated list of auto-annotation labels
		
		The input file should have one column per cluster and a layer named "trinaries"

		"""
		ds.ca.AutoAnnotation = self.fit(ds)
