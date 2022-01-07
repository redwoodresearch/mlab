import requests
from typing import Dict, List, Any
import numpy as np

TOKEN = "tao-1637086550-0-0a7d7030fbb8e316da58c6afce7c6315"
REPO_NAME = "redwoodresearch/mlab"
PATH_TO_MY_FILE = "days/w1d4/gorilla.py"
GIT_COMMIT = "df27e71724f32578c0921cfe8bc77ed2ec12b6df"
BRANCH_NAME = "tony"
API_URL = "https://jobs.redwoodresearchcompute.com:10101/api"
JOB_NAME = "mlab_w1d4_0_beth_tony"

json_dict = {
    "token": TOKEN,
    "kind": "submitJob",
    "name": JOB_NAME,
    "git_commit": GIT_COMMIT,
    "git_repo": REPO_NAME,
    "git_branch": BRANCH_NAME,
    "command": ["python", PATH_TO_MY_FILE],
    "scheduling_requirements": {
        "schedulability": True,
        "resource_options": [["A100"], ["RTX3090"], ["V100"]],
    },
}


def make_grid(possible_values: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    all_dicts = [dict({})]
    for name, vals in possible_values.items():
        tmp_dicts = []
        for val in vals:
            print(all_dicts)
            for d in all_dicts:
                new = d.copy()
                new.update({name: val})
                tmp_dicts.append(new)
        all_dicts = tmp_dicts
    return all_dicts


example = {
    "optim.Adam.lr": np.geomspace(1e-1, 1e-3, 3),
    "MyNet.H": [768, 1024],
}


def run(possible_values=example):
    gin_config_dicts = make_grid(possible_values)
    tasks = []

    for gin_config in gin_config_dicts:
        txt = ""
        for name, val in gin_config.items():
            txt += f"{name} = {val} \n "
        tasks.append(
            {
                "priority": 1,
                "parameters": {  # these are environment variables
                    "GIN_CONFIG": txt,
                    "RR_JOB":'True',
                },
            }
        )

    json_dict["tasks"] = tasks
    # print(json_dict)
    response = requests.post(API_URL, json=json_dict)
    print(response)


run()
