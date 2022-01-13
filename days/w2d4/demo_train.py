from comet_ml import Experiment
import torch as t
import torch.nn as nn
from sklearn.datasets import make_moons
import gin
from torch.utils.data import DataLoader, TensorDataset


@gin.configurable
class MyModel(nn.Module):
    def __init__(self, hidden_size, n_layers, input_size, output_size):
        super().__init__()
        self.mlps = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            *[nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)],
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.mlps(x)


@gin.configurable
def train(experiment, batch_size, lr, num_epochs):
    model = MyModel()
    optimizer = t.optim.Adam(model.parameters(), lr)
    dataset = t.rand(1000, batch_size, 2, 2)
    for epoch in range(num_epochs):
        for batch in dataset:
            loss = t.binary_cross_entropy_with_logits(model(batch[0]), batch[1]).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
