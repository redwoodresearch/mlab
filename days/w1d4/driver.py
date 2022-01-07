#!/usr/bin/env python3
import json
import sys
import requests
import subprocess

config_space = []

sizes_to_try = [
    [100],
    [50] * 2,
    [20] * 5,
    [10] * 10,
]

for lr in [1e-2]:
    for sizes in sizes_to_try:
        config_space.append(
            '\n'.join([
                f"RaichuModel.hidden_layer_sizes = {sizes}",
                f"Adam.lr = {lr}",
            ])
        )

def get_git_commit():
    return (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )

def task(config):
    return {
        "priority": 1,
        "parameters": {
            "gin_config": config,
        }
    }

query_json = {
    "token": "tao-1637086550-0-0a7d7030fbb8e316da58c6afce7c6315",
    "kind": "submitJob",
    "name": "basic-search",
    "git_commit": get_git_commit(),
    "git_repo": 'redwoodresearch/mlab',
    "git_branch": "bmillwood",
    "command": ["python", "days/w1d4/experiment.py"], 
    "tasks": [task(config) for config in config_space],
    "scheduling_requirements": {
        "schedulability": True,
        "resource_options": [["A100"], ["RTX3090"], ["V100"]]
    },
}

response = requests.post(
    'https://jobs.redwoodresearchcompute.com:10101/api',
    json=query_json
)

resp_json = json.loads(response.content)
print(resp_json)
print(f'https://jobs.redwoodresearchcompute.com/jobs/j{resp_json["job_id"]}')