import numpy as np
from datetime import datetime
import requests
import subprocess

def make_grid(possible_values):
    grid = []
    total_size = 1
    for values in possible_values.values():
        total_size *= len(values)
    for i in range(total_size):
        assignment = {}
        for name, values in possible_values.items():
            values_i = i % len(values)
            assignment[name] = values[values_i]
            i //= len(values)
        grid.append(assignment)
    return grid

def format_s(v):
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)

def job_str(job_dict):
    return "\n".join([f"{k}={v}" for k, v in job_dict.items()])

def write_gin_file(job_dict, filename):
    with open(filename, "w") as ginfile:
        ginfile.write(job_str(job_dict))
        
def make_grid_with_files(possible_values):
    grid = make_grid(possible_values)
    for i, job in enumerate(grid):
        filename = f"config_{i}.gin"
        write_gin_file(job, filename)

def get_git_commit():
    return (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )

def get_git_branch():
    return (
        subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )

def create_json(gin_config, filename):
    commit = get_git_commit()
    branch = get_git_branch()
    return {
        "token": "tao-1637086550-0-0a7d7030fbb8e316da58c6afce7c6315",
        "kind": "submitJob",
        "name": f"{filename}_{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}",
        "git_commit": commit,
        "git_repo": "redwoodresearch/mlab",
        "git_branch": branch,
        "command": ["python", filename], 
        "tasks": [{
            "priority": 1,
            "parameters": { "GIN_CONFIG": gin_config, }
        }],
        "scheduling_requirements": {"schedulability": True, "resource_options": [["A100"], ["RTX3090"], ["V100"]]},
    }


if __name__ == "__main__":
    grid_values = {
        "newtrain.learning_rate" : np.geomspace(1e-1, 1e-3, 2),
        "newtrain.momentum" :  [0.9],
        "newtrain.epochs" : [16],
        "newtrain.optimizer" : ["adam"],
        "newtrain.loss" : ["mse"],
        "newtrain.hidden_size" : [400, 800],
        "newtrain.weight_decay" : [0],
    }
    grid = make_grid(grid_values)
    for config in grid:
        js = job_str(config)
        json = create_json(config, "days/w1d4/picture_learning.py")
        print("sending job", json)
        response = requests.post("https://jobs.redwoodresearchcompute.com:10101/api", json=json)
        print(response)
        print("\n----------------------\n")
        print("breaking early so we don't spam the API")
        break
    