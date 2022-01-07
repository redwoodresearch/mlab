import comet_ml
from comet_ml import Experiment

import gin
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any
import numpy as np
import itertools


def make_grid(possible_values: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    out = []
    for combination in itertools.product(*possible_values.values()):
        out.append(dict(zip(possible_values.keys(), combination)))
    return out


def run_experiment(hyper_dict):    
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="OWGvZgoHXdqcGOgsFdDIeYHF8",
        project_name="general",
        workspace="maryphuong",
    )

    @gin.configurable
    class _MLP(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x):
            return self.layers(x)

    @gin.configurable
    def _train(model: nn.Module, dataloader: DataLoader, lr, momentum):
        opt = torch.optim.SGD(model.parameters(), lr, momentum)
        for X, y in dataloader:
            opt.zero_grad()
            pred = model(X)
            # print(pred.shape, y.shape)
            #assert False
            loss = F.l1_loss(pred, y)
            loss.backward()
            opt.step()
        return model


    def _evaluate(model: nn.Module, dataloader: DataLoader):
        sum_abs = 0.0
        n_elems = 0
        for X, y in dataloader:
            sum_abs += (model(X) - y).abs().sum()
            n_elems += y.shape[0] * y.shape[1]
        return sum_abs / n_elems


    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=int).unsqueeze(-1)

    # plt.scatter(X[:, 0], X[:, 1], c=y)

    dl = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)


    with gin.unlock_config():
        gin.parse_config_files_and_bindings(["config.gin"], 
                                            bindings=[f"{k}={v}" for k, v in hyper_dict.items()])

        for name, val in gin.get_bindings(_MLP).items():
            if type(val) in [str, int, float]:
                experiment.log_parameter('MLP.' + name, val)
        for name, val in gin.get_bindings(_train).items():
            if type(val) in [str, int, float]:
                experiment.log_parameter('train.' + name, val)

        # # Long any time-series metrics:
        # train_accuracy = 3.14
        # experiment.log_metric("accuracy", train_accuracy, step=0)

        model = _MLP()
        for i in range(200):
            model = _train(model, dl)
            eval_loss = _evaluate(model, dl)
            experiment.log_metric('l1_loss', eval_loss, step=i)

    experiment.end()

    
if __name__ == "__main__":
    hyper = {
        '_MLP.hidden_dim': [16, 32, 64],
        '_train.lr': [0.01, 0.02, 0.05, 0.1]
    }
    
    i = 0
    run_experiment(make_grid(hyper)[i])
