import itertools
import json
import os
import sys

from comet_ml import Experiment
import gin
import torch
import torch as t
from torch import nn
import torch.nn.functional as F

from days.rrjobs import rrjobs_submit

RR_API_KEY = "vABV7zo6pqS7lfzZBhyabU2Xe"

def make_grid(axes):
    value_sets = list(itertools.product(*axes.values()))
    points = []
    for item in value_sets:
        point_config = {}
        for i, key in enumerate(axes.keys()):
            point_config[key] = item[i]
        points.append(
            "\n".join([f"{key} = {repr(value)}" for key, value in point_config.items()])
        )
    return points

def get_cometml_api_key():
    api_key = None
    comet_api_file = os.path.expanduser('~/.comet-api-key')
    if os.path.exists(comet_api_file):
        with open(comet_api_file) as f:
            api_key = f.read().strip()
    else:
        print('Using RR Comet ML API key. Results link might be private.')
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
    from days.training_tests import load_image
    return load_image(image_name)[0]  # (train, test)


def search(name, params, location="local"):
    grid_points = make_grid(params)
    print(grid_points)
    if location == "remote":
        rrjobs_submit(
            name,
            command=["python", "days/hpsearch.py", "work"],
            tasks=[
                {"priority": 1, "parameters": {"gin_config": point}}
                for point in grid_points
            ],
        )
    else:
        for point in grid_points:
            gin.parse_config(point)
            train()


@gin.configurable
def train(optimizer='sgd', num_epochs=5, lr=1e-3):
    experiment = Experiment(
        project_name="project_name",
        api_key=get_cometml_api_key(),
    )  # it doesn't log parameters rn!
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
        print(loss)
    experiment.end()


if __name__ == "__main__":
    os.system("pip install -r requirements.txt") # server does not have all requirements
    if sys.argv[1] == "orchestrate":
        print("orchestrating")
        location = 'remote' if len(sys.argv) > 2 and sys.argv[2] == 'remote' else 'local'
        search(
            name="test_mlab_rrjobs_search",
            params={
                "train.lr": [0.001, 0.01],
                "train.optimizer": ["adam"],
                "train.num_epochs": [10, 20],
                "Model.hidden1": [10],
                "Model.hidden2": [20],
                "Model.hidden3": [30],
                "data_train.image_name": ["image_match1.png"],
            },
            location=location,
        )
    elif sys.argv[1] == "work":
        params = json.loads(os.environ["PARAMS"])
        gin.parse_config(params["gin_config"])
        train()
