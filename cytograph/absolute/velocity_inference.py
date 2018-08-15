import numpy as np
import torch
from torch.autograd import Variable
from sklearn.decomposition import PCA
import scipy.sparse as sparse
from tqdm import trange
from scipy.stats import linregress
from typing import *
from types import SimpleNamespace


class VelocityInference:
	def __init__(self, n_components: int = 10, lr: float = 0.0001, n_epochs: int = 10) -> None:
		self.lr = lr
		self.n_components = n_components

		self.n_epochs = n_epochs
		self.loss: float = 0
		self.a: np.ndarray = None
		self.b: np.ndarray = None
		self.g: np.ndarray = None

	def fit(self, s: np.ndarray, u: np.ndarray, knn: sparse.coo_matrix, g_init: np.ndarray = None) -> Any:
		"""
		Args:
			s      (n_genes, n_cells)
			u      (n_genes, n_cells)
			ds_du  (n_genes, n_cells)
		"""
		n_cells = s.shape[1]
		n_genes = s.shape[0]

		self.ds_du = np.zeros((n_genes, n_cells))
		for i in trange(n_cells):
			cells = (knn[i, :].A[0] > 0)
			sc = s[:, cells]
			uc = u[:, cells]
			for j in range(n_genes):
				if sc[j, :].sum() == 0 and uc[j, :].sum() == 0:
					self.ds_du[j, i] = 0
				else:
					with np.errstate(divide='ignore', invalid='ignore'):
						self.ds_du[j, i] = linregress(sc[j, :], uc[j, :]).slope
		self.ds_du = np.nan_to_num(self.ds_du, False)

		# Calculate the PCA transformation
		self.pca = PCA(n_components=self.n_components).fit(s)

		# Set up the optimization problem
		dt = torch.float
		if g_init is not None:
			g = Variable(torch.tensor(g_init, dtype=dt), requires_grad=True)
		else:
			g = Variable(0.1 * torch.ones(n_genes, dtype=dt), requires_grad=True)
		self.model = SimpleNamespace(
			sv=Variable(torch.tensor(s, dtype=dt)),
			uv=Variable(torch.tensor(u, dtype=dt)),
			k=Variable(torch.tensor(self.ds_du, dtype=dt)),
			pca_components=Variable(torch.tensor(self.pca.components_, dtype=dt)),
			pca_means=Variable(torch.tensor(self.pca.mean_, dtype=dt)),
			a_latent=Variable(torch.tensor(np.random.gamma(1, 1, size=(n_genes, self.n_components)), dtype=dt), requires_grad=True),
			b=Variable(torch.ones(n_genes, dtype=dt), requires_grad=True),
			g=g
		)
		self.epochs(self.n_epochs)
		return self

	def epochs(self, n_epochs: int):
		m = self.model
		loss_fn = torch.nn.MSELoss()
		optimizer = torch.optim.SGD([m.a, m.b, m.g], lr=self.lr)

		for epoch in trange(n_epochs):
			optimizer.zero_grad()
			u_pred = (m.a_latent @ m.pca_components + m.pca_means + m.g.unsqueeze(1) * m.k * m.sv) / (m.b.unsqueeze(1) * m.k + m.b.unsqueeze(1))
			loss_out = loss_fn(u_pred, m.uv)
			loss_out.backward()
			optimizer.step()
			a_latent.data.clamp_(min=0)
			b.data.clamp_(min=0)
			g.data.clamp_(min=0)

		self.loss = float(loss_out)
		self.a_latent = m.a_latent.detach().numpy()
		self.a = (m.a_latent @ m.pca_components + m.pca_means).detach().numpy()
		self.b = m.b.detach().numpy()
		self.g = m.g.detach().numpy()
		return self
