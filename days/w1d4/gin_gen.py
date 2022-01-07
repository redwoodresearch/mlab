from itertools import product
import numpy as np
import subprocess
import requests
import os

def make_grid(possible_values):
    value_combinations = product(*possible_values.values())
    return [
        dict(zip(possible_values.keys(), value_combo)) 
        for value_combo in value_combinations
    ]


search_space = {
    "train.lr": np.geomspace(1e-1, 1e-3, 2),
    # "run.hidden_size": [100, 400, 700, 1000]
    "run.hidden_size": [100, 200],
    "train.epochs": [10],
}

def make_request_dict(search_space):
    grid = make_grid(search_space)

    def make_task_dict(combo):
        gin_config = "\n".join([
            f"{k}={v}" for k, v in combo.items()
        ])
        return {
            'priority': 1,
            'parameters': {
                'gin_config': gin_config
            }
        }

    git_commit = (subprocess
        .check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )

    return {
        'token': "tao-1637086550-0-0a7d7030fbb8e316da58c6afce7c6315",
        "kind": "submitJob",
        "name": "nicholas and daniel w1d4 gridsearch",
        "git_commit": git_commit,
        "git_repo": "redwoodresearch/mlab",
        "command": ["python", "days/w1d4/w1d4.py"],
        "tasks": [make_task_dict(x) for x in grid],
        "scheduling_requirements": {
            "schedulability": True,
            "resource_options": [["A100"], ["RTX3090"], ["V100"]]
        }
    }

def run_local(search_space):
    req = make_request_dict(search_space)
    for task in req["tasks"]:
        os.environ["PARAMS"] = str(task["parameters"])
        subprocess.run(["python3", "w1d4.py"])


requests.post(
    "https://jobs.redwoodresearchcompute.com:10101/api",
    json=make_request_dict(search_space)
)
# run_local(search_space)