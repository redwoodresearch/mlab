from itertools import product
import numpy as np

def make_grid(possible_values):
    value_combinations = product(*possible_values.values())
    return [
        dict(zip(possible_values.keys(), value_combo)) 
        for value_combo in value_combinations
    ]


search_space = {
    "train.lr": np.geomspace(1e-1, 1e-3, 5),
    "run.hidden_size": [100, 400, 700, 1000]
}

def make_request(search_space):
    grid = make_grid(search_space)
    gin_config = "\n".join([
        f"{k}={v}" for k, v in grid.items()
    ])
    return {
        'token': "tao-1637086550-0-0a7d7030fbb8e316da58c6afce7c6315",
        "kind": "submitJob",
        "name": "nicholas and daniel w1d4 gridsearch",
        "git_commit": "",
        "git_repo": "redwoodresearch/mlab",
        "command": ["python", "days/w1d4/w1d4.py"],
        "tasks": [{
            'priority': 1,
            'parameters': {
                'gin_config': gin_config
            }
        }],
        "scheduling_requirements": {
            "schedulability": True,
            "resource_options": [["A100"], ["RTX3090"], ["V100"]]
        }
    }
