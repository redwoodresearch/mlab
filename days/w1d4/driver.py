import sys
import requests

config_space = []

for lr in [1e-2]:
    for H in list(range(100, 501, 50)): ## [100, 150, 200, 250, 300]:
        config_space.append(
            '\n'.join([
                f"RaichuModel.H = {H}",
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

json = {
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
    json=json
)

print(response.content)
