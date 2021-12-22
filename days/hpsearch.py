import comet_ml
from comet_ml import Experiment
import sys
import json

comet_ml.init()
import gin
import torch as t
from torch import einsum
from torch import nn
import math
import matplotlib.pyplot as plt
import itertools as pr
from days.rrjobs import rrjobs_submit
import torch.nn.functional as F


def make_grid(axes):
    value_sets = list(pr.product(*axes.values()))
    points = []
    for item in value_sets:
        point_config = {}
        for i, key in enumerate(axes.keys()):
            point_config[key] = item[i]
        points.append(point_config)
    return points


@gin.configurable
class Model(nn.Module):
    def __init__(self, P, H, K):
        super().__init__()
        self.P = nn.Linear(2, P)
        self.H = nn.Linear(P, H)
        self.K = nn.Linear(H, K)
        self.P = nn.Linear(K, 3)

    def forward(self, x):
        return self.O(F.gelu(self.K(F.gelu(self.H(F.gelu(self.P(x)))))))


def data_train():
    import days.training_tests as tt

    return tt.load_image()


def search(name, axes, location="remote"):
    grid_points = make_grid(axes)
    if location == "remote":
        rrjobs_submit(
            name,
            ["python", "days/hpsearch.py", "work"],
            [
                {"priority": 1, "parameters": {"--grid_point": json.dumps(point)}}
                for point in grid_points
            ],
        )
    else:
        for point in grid_points:
            with gin.unlock_config():
                gin.parse_config_files_and_bindings([], point)
            train()


@gin.configurable
def train(optimizer, num_epochs, lr, project_name):
    experiment = Experiment(project_name="project_name")
    model = Model()
    params = list(model.parameters())
    if optimizer == "sgd":
        optimizer = t.optim.SGD(params, lr=lr)
    elif optimizer == "rmsprop":
        optimizer = t.optim.RMSprop(params, lr=lr)
    elif optimizer == "adam":
        optimizer = t.optim.Adam(params, lr=lr)
    loss_fn = t.nn.L1Loss()
    dataloader = data_train()
    for _ in range(num_epochs):
        for input, target in dataloader:
            optimizer.zero_grad()  # sets param.grad to None for linked params
            # (this prevents accumulation of gradients)
            output = model(input)  # the model’s predictions
            loss = loss_fn(output, target)  # measure how bad predictions are
            loss.backward()  # calculate gradients
            optimizer.step()  # use gradients to update params
    experiment.end()


if __name__ == "__main__":
    if sys.argv[1] == "orchestrate":
        search({})
    elif sys.argv[1] == "work":
        gin.parse_config_files_and_bindings([], json.loads(sys.argv[3]))
        train()
