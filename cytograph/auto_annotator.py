import os
import logging
import numpy as np


class CellTag:
	def __init__(self, category, file):
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
		if not hasattr(self, "name"):
			raise ValueError("'name' was missing")
		if not hasattr(self, "abbreviation"):
			raise ValueError("'abbreviation' was missing")
		if not hasattr(self, "positives"):
			raise ValueError("positive markers were missing")
		if not hasattr(self, "negatives"):
			self.negatives = []

	def __str__(self):
		temp = self.name + " (" + self.abbreviation + "; " + " ".join(["+" + x for x in self.positives])
		if len(self.negatives) > 0:
			temp = temp  + " " + " ".join(["-" + x for x in self.negatives]) + ")"
		else:
			temp = temp + ")"
		return temp

class AutoAnnotator(object):
	def __init__(self, ds, root="/Users/Sten/Dropbox (Linnarsson Group)/Code/autoannotation/"):
		root_len = len(root)
		self.tags = []
		errors = False
		for cur, dirs, files in os.walk(root):
			for file in files:
				if file[-3:] == ".md" and file[-9:] != "README.md":
					try:
						tag = CellTag(cur[root_len:], os.path.join(cur, file))
						for pos in tag.positives:
							if not ds.Gene.__contains__(pos):
								logging.error(file + ": gene '%s' not found in file", pos)
								errors = True
						for neg in tag.negatives:
							if not ds.Gene.__contains__(neg):
								logging.error(file + ": gene '%s' not found in file", neg)
								errors = True
						self.tags.append(tag)
					except ValueError as e:
						logging.error(file + ": " + str(e))
						errors = True
		if errors:
			raise ValueError("Error loading cell tag definitions")

	def annotate(self, ds, trinaries):
		"""
		Annotate the dataset based on trinarized gene expression
		"""
		annotations = np.empty((len(self.tags), trinaries.shape[1]))
		for ix, tag in enumerate(self.tags):
			for cluster in range(trinaries.shape[1]):
				p = 1
				for pos in tag.positives:
					index = np.where(ds.Gene == pos)[0][0]
					p = p*trinaries[index, cluster]
				for neg in tag.negatives:
					index = np.where(ds.Gene == neg)[0][0]
					p = p*(1-trinaries[index, cluster])
				annotations[ix, cluster] = p
		return (self.tags, annotations)
