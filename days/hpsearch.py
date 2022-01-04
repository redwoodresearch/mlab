import itertools
import json
import os
import sys
from pprint import pprint

# server is missing requirements
# this call has to happen before imports
os.system("pip install -q -r requirements.txt")

from comet_ml import Experiment
import gin
import numpy as np
import torch
import torch as t
from torch import nn
import torch.nn.functional as F

from days.rrjobs import rrjobs_submit

RR_API_KEY = "vABV7zo6pqS7lfzZBhyabU2Xe"


def make_grid(axes):
    return [
        {key: value for key, value in zip(axes.keys(), values_choice)}
        for values_choice in itertools.product(*axes.values())
    ]


def to_gin_config_str(params):
    return "\n".join(f"{key} = {repr(value)}" for key, value in params.items())


def get_cometml_api_key():
    api_key = None
    comet_api_file = os.path.expanduser("~/.comet-api-key")
    if os.path.exists(comet_api_file):
        with open(comet_api_file) as f:
            api_key = f.read().strip()
    else:
        print(
            "Using RR Comet ML API key. Results link might be private. You may instead "
            + "add your key to `~/.comet-api-key`."
        )
        api_key = RR_API_KEY
    return api_key


@gin.configurable
class Model(nn.Module):
    def __init__(self, hidden1=10, hidden2=10, hidden3=10):
        super().__init__()
        n_coords = 2
        n_colors = 3
        self.net = nn.Sequential(
            nn.Linear(n_coords, hidden1),
            nn.GELU(),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Linear(hidden2, hidden3),
            nn.GELU(),
            nn.Linear(hidden3, n_colors),
        )

    def forward(self, x):
        return self.net(x)


@gin.configurable
def data_train(image_name):
    from days.w1d5.w1d5_tests import load_image

    return load_image(image_name)[0]  # (train, test)


@gin.configurable
def train(optimizer="sgd", num_epochs=5, lr=1e-3):
    model = Model()

    optimizer_fns = {
        "sgd": torch.optim.SGD,
        "rmsprop": t.optim.RMSprop,
        "adam": t.optim.Adam,
    }
    optimizer_fn = optimizer_fns[optimizer]
    optimizer = optimizer_fn(model.parameters(), lr=lr)

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
        print(f"loss: {loss.item()}")


def run_experiment(project_name, hyperparameters):
    gin_config_str = to_gin_config_str(hyperparameters)
    gin.parse_config(gin_config_str)

    print("Running experiment with hyperparameters:")
    pprint(hyperparameters)

    experiment = Experiment(
        project_name=project_name,
        api_key=get_cometml_api_key(),
    )
    experiment.log_parameters(hyperparameters)
    train()
    experiment.end()


if __name__ == "__main__":
    if sys.argv[1] == "orchestrate":
        is_remote = len(sys.argv) > 2 and sys.argv[2] == "remote"

        project_name = f"hpsearch-{np.random.randint(10000)}"

        params = {
            "train.lr": [0.001, 0.01],
            "train.optimizer": ["adam"],
            "train.num_epochs": [10, 20],
            "Model.hidden1": [10],
            "Model.hidden2": [20],
            "Model.hidden3": [30],
            "data_train.image_name": ["image_match1.png"],
        }
        hyperparameter_grid = make_grid(params)

        if is_remote:
            rrjobs_submit(
                name="test_mlab_rrjobs_search",
                command=["python", "days/hpsearch.py", "experiment"],
                tasks=[
                    {
                        "priority": 1,
                        "parameters": {
                            "project_name": project_name,
                            "hyperparameters": hyperparameters,
                        },
                    }
                    for hyperparameters in hyperparameter_grid
                ],
            )
        else:
            for hyperparameters in hyperparameter_grid:
                run_experiment(project_name, hyperparameters)

    elif sys.argv[1] == "experiment":
        params = json.loads(os.environ["PARAMS"])
        project_name = params["project_name"]
        hyperparameters = params["hyperparameters"]
        run_experiment(project_name, hyperparameters)
