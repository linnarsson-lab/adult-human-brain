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

	def fit(self, X: sparse.coo_matrix, test_size: float = 0.25, validate_size: float = 0.25) -> None:
		"""
		Fit a BNPF model to the data matrix

		Args:
			x      			Data matrix, shape (n_samples, n_features)
			test_size		Fraction of the data to use for testing the performance, in (0, 1)
			validate_size	Fraction of the data to use for validating the performance, in (0, 1)

		Returns:
			self
		
		Remarks:
			After fitting, the factor matrices W and H are available as self.W of shape
			(n_samples, n_components) and self.H of shape (n_components, n_features)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		# Split the input
		# Note: gaprec wants the same matrix for training, test and validation, but with subsets of the nonzeros held out; hence we need to split the underlying nonzeros
		rest_row, rest_col, rest_data, test_row, test_col, test_data = train_test_split(X.row, X.col, X.data, test_size=test_size)
		train_row, train_col, train_data, validate_row, validate_col, validate_data = train_test_split(rest_row, rest_col, rest_data, test_size=(validate_size / (1 - test_size)))
		
		# Save to TSV files
		def save_tsv(fname: str, m: sparse.coo_matrix) -> None:
			b = m.tocoo()
			np.savetxt("test.tsv", np.vstack([b.row, b.col, b.data]).T, delimiter="\t")
		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpdirname = "/Users/sten/gaprec"
			if not os.path.exists(tmpdirname):
				os.mkdir(tmpdirname)
			save_tsv(os.path.join(tmpdirname, "train.tsv"), sparse.coo_matrix((train_data, (train_row, train_col)), shape=X.shape))
			save_tsv(os.path.join(tmpdirname, "test.tsv"), sparse.coo_matrix((test_data, (test_row, test_col)), shape=X.shape))
			save_tsv(os.path.join(tmpdirname, "validation.tsv"), sparse.coo_matrix((validate_data, (validate_row, validate_col)), shape=X.shape))

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
