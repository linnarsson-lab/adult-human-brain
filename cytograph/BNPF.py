from typing import *
import tempfile
import os
from subprocess import Popen
import numpy as np
import logging
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split


class BNPF:
	def __init__(self, max_components: int, alpha: float = 1.1, c: float = 1) -> None:
		self.max_components = max_components
		self.alpha = alpha
		self.c = c
		self.W: np.ndarray = None
		self.H: np.ndarray = None

	def fit(self, X: sparse.coo_matrix) -> None:
		"""
		Fit a BNPF model to the data matrix

		Args:
			x      			Data matrix, shape (n_samples, n_features)

		Returns:
			self
		
		Remarks:
			After fitting, the factor matrices W and H are available as self.W of shape
			(n_samples, n_components) and self.H of shape (n_components, n_features)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpdirname = "/Users/sten/gaprec"
			if not os.path.exists(tmpdirname):
				os.mkdir(tmpdirname)
			# Save to TSV file
			np.savetxt(os.path.join(tmpdirname, "train.tsv"), np.vstack([X.row, X.col, X.data]).T, delimiter="\t")
			np.savetxt(os.path.join(tmpdirname, "test.tsv"), np.vstack([X.row, X.col, X.data]).T, delimiter="\t")
			np.savetxt(os.path.join(tmpdirname, "validation.tsv"), np.vstack([X.row, X.col, X.data]).T, delimiter="\t")

			# Run gaprec
			bnpf_p = Popen((
				"gaprec",
				"-dir", tmpdirname,
				"-m", str(X.shape[0]),
				"-n", str(X.shape[1]),
				"-T", str(self.max_components),
				"-alpha", str(self.alpha),
				"-C", str(self.c)
			), cwd=tmpdirname)
			bnpf_p.wait()
			if bnpf_p.returncode != 0:
				logging.error("BNPF failed to execute external binary 'gaprec' (check $PATH)")
				raise RuntimeError()
