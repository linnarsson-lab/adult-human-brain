from typing import *
import tempfile
import os
from subprocess import Popen
import numpy as np
import logging
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split


class HPF:
	def __init__(self, k: int, a: float = 0.3, b: float = 0.3, c: float = 0.3, d: float = 0.3) -> None:
		self.k = k
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.beta: np.ndarray = None
		self.theta: np.ndarray = None

	def fit(self, X: sparse.coo_matrix, test_size: float = 0.25, validate_size: float = 0.25) -> None:
		"""
		Fit a BNPF model to the data matrix

		Args:
			x      			Data matrix, shape (n_samples, n_features)
			test_size       Fraction to use for test dataset, or None to use X for training, test and validation
			validation_size Fraction to use for validation dataset

		Returns:
			self
		
		Remarks:
			After fitting, the factor matrices W and H are available as self.W of shape
			(n_samples, n_components) and self.H of shape (n_components, n_features)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		# Split the input
		# Note: hgaprec wants the same matrix for training, test and validation, but with subsets of the nonzeros held out; hence we need to split the underlying nonzeros
		if test_size is not None:
			rest_row, test_row, rest_col, test_col, rest_data, test_data = train_test_split(X.row, X.col, X.data, test_size=test_size)
			train_row, validate_row, train_col, validate_col, train_data, validate_data = train_test_split(rest_row, rest_col, rest_data, test_size=(validate_size / (1 - test_size)))

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpdirname = "/Users/sten/gaprec"
			if not os.path.exists(tmpdirname):
				os.mkdir(tmpdirname)
			# Save to TSV file
			np.set_printoptions(precision=1, suppress=True)
			
			if test_size is not None:
				np.savetxt(os.path.join(tmpdirname, "train.tsv"), np.vstack([train_row, train_col, train_data]).T, delimiter="\t", fmt="%d")
				np.savetxt(os.path.join(tmpdirname, "test.tsv"), np.vstack([test_row, test_col, test_data]).T, delimiter="\t", fmt="%d")
				np.savetxt(os.path.join(tmpdirname, "validation.tsv"), np.vstack([validate_row, validate_col, validate_data]).T, delimiter="\t", fmt="%d")
			else:
				np.savetxt(os.path.join(tmpdirname, "train.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
				np.savetxt(os.path.join(tmpdirname, "test.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
				np.savetxt(os.path.join(tmpdirname, "validation.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")

			# Run hgaprec
			# hgaprec -dir ~/midbrain -n 2011 -m 986 -k 10 -label hi -hier
			bnpf_p = Popen((
				"hgaprec",
				"-dir", tmpdirname,
				"-m", str(X.shape[1]),
				"-n", str(X.shape[0]),
				"-k", str(self.k),
				"-a", str(self.a),
				"-b", str(self.b),
				"-c", str(self.c),
				"-d", str(self.d),
				"-hier"
			), cwd=tmpdirname)
			bnpf_p.wait()
			if bnpf_p.returncode != 0:
				logging.error("BNPF failed to execute external binary 'gaprec' (check $PATH)")
				raise RuntimeError()
			sf = f"n{X.shape[0]}-m{X.shape[1]}-k{self.k}-batch-hier-vb"

			# Format of these is (row, col, )
			self.theta = np.loadtxt(os.path.join(tmpdirname, sf, "htheta.tsv"))[:, 2:]
			temp = np.loadtxt(os.path.join(tmpdirname, sf, "hbeta.tsv"))
			self.beta = temp[:, 2:][np.argsort(temp[:, 1]), :]  # the beta matrix with the correct rows ordering
