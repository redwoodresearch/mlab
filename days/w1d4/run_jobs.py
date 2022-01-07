import requests
import subprocess
import numpy as np
from itertools import product


def make_grid(possible_values):
    all_combinations = product(
        *[[(key, value) for value in values] for key, values in possible_values.items()]
    )
    return list(map(dict, all_combinations))


hyper_parameters = make_grid(
    {
        "train.lr": np.geomspace(1e-1, 1e-4, 3),
        "model.hidden_size": [768, 1024],
        "train.epoch": [10, 50],
        "train.optimizer": ["adam", "sgd", "rmsprop"],
        "train.weight_decay": [0.0, 0.05],
        "train.momentum": [0.5, 0.9],
    }
)


def get_git_commit():
    return (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )


def make_tasks():
    for hyper_params in hyper_parameters:

        params = f"""train.epochs = 50
    train.optimizer = "sgd"
    train.lr = {hyper_params['train.lr']}
    train.weight_decay = 0.05
    train.momentum = 0.9
    train.dampening = 0
    model.hidden_size = {hyper_params['model.hidden_size']}"""
        yield {
            "priority": 1,
            "parameters": {"GIN_CONFIG": params},  # these are environment variables
        }


data = {
    "token": "tao-1637086550-0-0a7d7030fbb8e316da58c6afce7c6315",
    "kind": "submitJob",
    "name": "tim-hyperparameter-search",
    "git_commit": get_git_commit(),
    "git_repo": "redwoodresearch/mlab",
    "git_branch": "alwin",
    "command": ["python", "days/w1d4/training.py"],
    "tasks": list(make_tasks()),
    "scheduling_requirements": {
        "schedulability": True,
        "resource_options": [["A100"], ["RTX3090"], ["V100"]],
    },
}

API_URL = "https://jobs.redwoodresearchcompute.com:10101/api"

requests.post(API_URL, json=data)
