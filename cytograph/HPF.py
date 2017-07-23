from typing import *
import tempfile
import os
from subprocess import Popen
import numpy as np
import logging
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError


class HPF:
	"""
	Bayesian scalable hierarchical Poisson matrix factorization
	See https://arxiv.org/pdf/1311.1704.pdf

	This implementation requires https://github.com/linnarsson-lab/hgaprec to be installed and in the $PATH
	"""
	def __init__(self, k: int, a: float = 0.3, b: float = 0.3, c: float = 0.3, d: float = 0.3) -> None:
		self.k = k
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.beta: np.ndarray = None
		self.theta: np.ndarray = None

		self._cache: Dict[str, str] = {}

	def fit(self, X: sparse.coo_matrix) -> None:
		"""
		Fit an HPF model to the data matrix

		Args:
			X      			Data matrix, shape (n_samples, n_features)
			test_size       Fraction to use for test dataset, or None to use X for training, test and validation
			validation_size Fraction to use for validation dataset

		Remarks:
			After fitting, the factor matrices beta and theta are available as self.beta of shape
			(n_samples, k) and self.theta of shape (k, n_features)
		"""
		if type(X) is not sparse.coo_matrix:
			raise TypeError("Input matrix must be in sparse.coo_matrix format")

		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpdirname = "/Users/sten/gaprec"
			if not os.path.exists(tmpdirname):
				os.mkdir(tmpdirname)
			# Save to TSV file
			np.savetxt(os.path.join(tmpdirname, "train.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
			np.savetxt(os.path.join(tmpdirname, "test.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
			np.savetxt(os.path.join(tmpdirname, "validation.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")

			# Run hgaprec
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
				logging.error("HPF failed to execute external binary 'hgaprec' (check $PATH)")
				raise RuntimeError()
			sf = f"n{X.shape[0]}-m{X.shape[1]}-k{self.k}-batch-hier-vb"

			# Format of these is (row, col, )
			self.theta = np.loadtxt(os.path.join(tmpdirname, sf, "htheta.tsv"))[:, 2:]
			temp = np.loadtxt(os.path.join(tmpdirname, sf, "hbeta.tsv"))
			self.beta = temp[:, 2:][np.argsort(temp[:, 1]), :]  # the beta matrix with the correct rows ordering

			# Cache the beta branch of the model, so we can refit later with fixed beta
			for file in ["betarate.tsv", "betarate_rate.tsv", "betarate_shape.tsv", "hbeta.tsv", "hbeta_rate.tsv", "hbeta_shape.tsv"]:
				with open(os.path.join(tmpdirname, sf, file)) as f:
					self._cache[file] = f.read()

	def transform(self, X: sparse.coo_matrix) -> np.ndarray:
		if self.beta is None:
			raise NotFittedError("Cannot transform without first fitting the model")
		if X.shape[1] != self.beta.shape[0]:
			raise ValueError(f"X must have exactly {self.beta.shape[0]} columns")
		if np.any(np.sum(X, axis=0) == 0):
			raise ValueError("Every feature (column) must have at least one non-zero sample (row)")
		if np.any(np.sum(X, axis=1) == 0):
			raise ValueError("Every sample (row) must have at least one non-zero feature (column)")
		with tempfile.TemporaryDirectory() as tmpdirname:
			tmpdirname = "/Users/sten/gaprec"
			if not os.path.exists(tmpdirname):
				os.mkdir(tmpdirname)

			# Save to TSV file
			np.savetxt(os.path.join(tmpdirname, "train.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
			np.savetxt(os.path.join(tmpdirname, "test.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")
			np.savetxt(os.path.join(tmpdirname, "validation.tsv"), np.vstack([X.row + 1, X.col + 1, X.data]).T, delimiter="\t", fmt="%d")

			# Write the previously saved beta samples for reuse
			# sf = f"n{X.shape[0]}-m{X.shape[1]}-k{self.k}-batch-hier-vb-beta-precomputed"
			# if not os.path.exists(os.path.join(tmpdirname, sf)):
			# 	os.mkdir(os.path.join(tmpdirname, sf))
				
			for file in self._cache.keys():
				with open(os.path.join(tmpdirname, file), "w") as f:
					f.write(self._cache[file])

			# Run hgaprec
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
				"-hier",
				"-beta-precomputed"
			), cwd=tmpdirname)
			bnpf_p.wait()
			if bnpf_p.returncode != 0:
				logging.error("HPF failed to execute external binary 'hgaprec' (check $PATH)")
				raise RuntimeError()
			sf = f"n{X.shape[0]}-m{X.shape[1]}-k{self.k}-batch-hier-vb-beta-precomputed"
			return np.loadtxt(os.path.join(tmpdirname, sf, "htheta.tsv"))[:, 2:]
