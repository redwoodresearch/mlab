import os

os.system("pip install -r requirements.txt")

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

tmp_gin_fname = ".hpsearch_temp_gin"


def make_grid(axes):
    value_sets = list(pr.product(*axes.values()))
    points = []
    for item in value_sets:
        point_config = {}
        for i, key in enumerate(axes.keys()):
            point_config[key] = item[i]
        points.append(
            "\n".join([f"{key} = {repr(value)}" for key, value in point_config.items()])
        )
    return points


@gin.configurable
class Model(nn.Module):
    def __init__(self, P, H, K):
        super().__init__()
        self.P = nn.Linear(2, P)
        self.H = nn.Linear(P, H)
        self.K = nn.Linear(H, K)
        self.O = nn.Linear(K, 3)

    def forward(self, x):
        return self.O(F.gelu(self.K(F.gelu(self.H(F.gelu(self.P(x)))))))


def data_train():
    import days.training_tests as tt

    return tt.load_image("image_match1.png")[0]  # (train, test)


def search(name, axes, location="local"):
    grid_points = make_grid(axes)
    print(grid_points)
    if location == "remote":
        rrjobs_submit(
            name,
            ["python", "days/hpsearch.py", "work"],
            [
                {"priority": 1, "parameters": {"gin_config": point}}
                for point in grid_points
            ],
        )
    else:
        for point in grid_points:
            with gin.unlock_config():
                with open(tmp_gin_fname, "w") as f:
                    f.write(point)
                gin.parse_config_file(tmp_gin_fname)
            train()


@gin.configurable
def train(optimizer, num_epochs, lr):
    experiment = Experiment(
        project_name="project_name",
        api_key="vABV7zo6pqS7lfzZBhyabU2Xe",
    )
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
        print(loss)
    experiment.end()


if __name__ == "__main__":
    if sys.argv[1] == "orchestrate":
        print("orchestrateing")
        search(
            "test_mlab_rrjobs_search",
            {
                "train.lr": [0.001, 0.01],
                "train.optimizer": ["adam"],
                "train.num_epochs": [10, 20],
                "Model.K": [10],
                "Model.H": [20],
                "Model.P": [30],
            },
            "remote",
        )
    elif sys.argv[1] == "work":
        params = json.loads(os.environ["PARAMS"])
        with open(tmp_gin_fname, "w") as f:
            f.write(params["gin_config"])
        gin.parse_config_file(tmp_gin_fname)
        train()
