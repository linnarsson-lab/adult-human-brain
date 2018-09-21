import numpy as np
import torch
from torch.autograd import Variable
from sklearn.decomposition import PCA
import scipy.sparse as sparse
from tqdm import trange
from scipy.stats import linregress
from typing import *
from types import SimpleNamespace
import loompy
import logging


class VelocityInference:
	def __init__(self, n_components: int = 10, lr: float = 0.0001, n_epochs: int = 10) -> None:
		self.lr = lr
		self.n_components = n_components

		self.n_epochs = n_epochs
		self.loss: float = 0
		self.a: np.ndarray = None
		self.b: np.ndarray = None
		self.g: np.ndarray = None

	def fit(self, ds: loompy.LoomConnection, g_init: np.ndarray = None) -> Any:
		"""
		Args:
			s      (n_genes, n_cells)
			u      (n_genes, n_cells)
		"""
		n_cells = ds.shape[1]
		logging.info("Determining selected genes")
		selected = (ds.ra.Selected == 1)
		n_genes = selected.sum()
		n_components = ds.ca.HPF.shape[1]
		logging.info("Loading spliced")
		s_data = ds["spliced"][:,:][selected, :].astype("float")
		logging.info("Loading unspliced")
		u_data = ds["unspliced"][:,:][selected, :].astype("float")
		logging.info("Loading HPF")
		m_data = ds.ra.HPF[selected, :]
		
		# Set up the optimization problem
		logging.info("Setting up the optimization problem")
		dt = torch.float
		if g_init is not None:
			g = Variable(torch.tensor(g_init, dtype=dt), requires_grad=True)
		else:
			g = Variable(0.1 * torch.ones(n_genes, dtype=dt), requires_grad=True)
		self.model = SimpleNamespace(
			s=Variable(torch.tensor(s_data, dtype=dt)),
			u=Variable(torch.tensor(u_data, dtype=dt)),
			m=Variable(torch.tensor(m_data, dtype=dt)),
			v=Variable(torch.tensor(np.random.normal(0, 1, size=(n_components, n_cells)), dtype=dt), requires_grad=True),
			b=Variable(torch.ones(n_genes, dtype=dt), requires_grad=True),
			g=g
		)
		logging.info("Optimizing")
		self.epochs(self.n_epochs)
		return self

	def epochs(self, n_epochs: int) -> Any:
		m = self.model
		loss_fn = torch.nn.MSELoss()
		optimizer = torch.optim.SGD([m.v, m.b, m.g], lr=self.lr)

		for epoch in trange(n_epochs):
			optimizer.zero_grad()
			left = m.m @ m.v
			right = m.b.unsqueeze(1) * m.u - m.g.unsqueeze(1) * m.s - m.m @ m.v
			loss_out = loss_fn(left, right)
			loss_out.backward()
			optimizer.step()
			m.b.data.clamp_(min=0)
			m.g.data.clamp_(min=0)

		self.loss = float(loss_out)
		self.v = m.v.detach().numpy()
		self.b = m.b.detach().numpy()
		self.g = m.g.detach().numpy()
		return self

