

class LassoRegressionModel(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, lr: float = 0.01, l1_weight: float = 1) -> None:
        super(LassoRegressionModel, self).__init__()

        # Parameters
        self.l1_weight = l1_weight

        # The neural network
        self.layer1 = nn.Linear(n_inputs, n_outputs)
        
        # The optimization algorithm
        self.optimiser = torch.optim.SGD(self.parameters(), lr = lr)

    def loss(self, y, y_pred):
        l1_reg = Variable(torch.FloatTensor(1), requires_grad=True)
        for W in self.parameters():
            l1_reg = l1_reg + W.norm(1)
        return nn.MSELoss()(y, y_pred) + self.l1_weight * l1_reg

    def forward(self, x):
        y = self.layer1(x)
        return y


class LinearRegressionModel(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, lr: float = 0.01) -> None:
        super(LinearRegressionModel, self).__init__()

        # The neural network
        self.layer1 = nn.Linear(n_inputs, n_outputs)

        # The optimization algorithm
        self.optimiser = torch.optim.SGD(self.parameters(), lr = lr)

    def loss(self, y, y_pred):
        return nn.MSELoss()(y, y_pred)

    def forward(self, x):
        y = self.layer1(x)
        return y

class WaddingtonianDataset:
    def __init__(self, ds: loompy.LoomConnection) -> None:
        human_TFs = pd.read_csv("/Users/stelin/code/development-human/TFs_human.txt", sep="\t").values.T[0]
        self.n_cells = ds.shape[1]
        self.regulators = np.isin(ds.ra.Gene, human_TFs)
        self.regulator_names = ds.ra.Gene[self.regulators]
        self.n_regulators = self.regulators.shape[0]
        self.targets = ds.ra.Selected == 1
        self.target_names = ds.ra.Gene[self.targets]
        self.n_targets = self.targets.shape[0]

        # Load the velocity data for all selected genes
        v_data = np.empty((self.n_targets, self.n_cells), dtype='float32')
        r = 0
        for (_, _, view) in ds.scan(items=self.targets, axis=0, layers=["spliced_exp", "velocity"]):
            v_data[r:r + view.shape[0], :] = view.layers["velocity"][:, :]
            r += view.shape[0]
        self.v = torch.from_numpy(v_data.T)

        # Load the expression data for TFs only
        s_data = np.empty((self.n_regulators, self.n_cells), dtype='float32')
        r = 0
        for (_, _, view) in ds.scan(items=self.regulators, axis=0, layers=["spliced_exp", "velocity"]):
            s_data[r:r + view.shape[0], :] = view.layers["spliced_exp"][:, :]
            r += view.shape[0]
        self.s = torch.from_numpy(s_data.T)
        
class WaddingtonianInference:
    def __init__(self, dataset: WaddingtonianDataset, model_type: nn.Module = LinearRegressionModel, lr: float = 0.001, n_epochs: int = 100, **kwargs) -> None:
        self.dataset = dataset
        self.model_type = model_type
        self.model: nn.Module = None
        self.model_kwargs = kwargs
        self.lr = lr
        self.n_epochs = n_epochs
        self.losses: List[float] = []

    def fit(self) -> None:
        self.model = self.model_type(self.dataset.n_regulators, self.dataset.n_targets, self.lr, **self.model_kwargs)

        t = trange(self.n_epochs)
        for epoch in t:
            self.model.optimiser.zero_grad()

            x = Variable(self.dataset.s)
            y = Variable(self.dataset.v)

            y_pred = self.model.forward(x)
            loss = self.model.loss(y, y_pred)
            self.losses.append(float(loss))
            loss.backward()
            self.model.optimiser.step()
            t.set_description(f"loss={float(loss)}")
            t.refresh()