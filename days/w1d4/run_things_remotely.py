import subprocess
import requests

def make_tasks(possible_values):
    size = reduce(mul, map(len, possible_values.values()))
    ls = []
    for idx in range(size):
        new_values = {}
        for name, values in possible_values.items():
            new_values[name] = values[idx % len(values)]
            idx //= len(values)
        ls.append({"priority": 1, "parameters": new_values})
    return ls

possible_values = {
    "gin_train.lr": [0.01, 0.001],
    "Model.P": [2],
    "Model.H": [100],
    "Model.K": [3],
    "gin_train.betas": [(0.9, 0.999)],
    "gin_config": ["days/w1d4/config.gin"]
}

def get_git_commit():
    return (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )

json_dict = {
    "token": "tao-1637086550-0-0a7d7030fbb8e316da58c6afce7c6315",
    "kind": "submitJob",
    "name": "eric_guilhermo_test1",
    "git_commit": get_git_commit(),
    "git_repo": "redwoodresearch/mlab",
    "git_branch": "eric_guilhermo",
    "command": ["python", "days/w1d4/whatever.py"], 
    "tasks": make_tasks(possible_values),
    # "tasks": [{
    #     "priority": 1,
    #     "parameters": { # these are environment variables
    #         "gin_config": "days/w1d4/config.gin",
    # }}],
    "scheduling_requirements": {"schedulability": True, "resource_options": [["A100"], ["RTX3090"], ["V100"]]},
}

response = requests.post("https://jobs.redwoodresearchcompute.com:10101/api", json=json_dict)
print(response)