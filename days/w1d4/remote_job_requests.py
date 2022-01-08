import requests
from gin_exercise import make_grid


# Figure out what the json_dicts we should feed in are

if __name__ == "__main__":
    possible_values = {"Model.hidden_size" : [256, 512],
                    "Model.num_hidden_layers": [1],
                }

    json_dict = {
        "token": "tao-1637086550-0-0a7d7030fbb8e316da58c6afce7c6315",
        "kind": "submitJob",
        "name": "decadent_cactus",
        "git_commit": "587c875187cfbeee93b1a6a8cc61446145814604",
        "git_repo": "redwoodresearch/mlab",
        "git_branch": "yulong_mia",
        "command": ["python", "days/w1d4/remote_job_script.py"],
        "tasks": [],
        "scheduling_requirements": {"schedulability": True, "resource_options": [["A100"], ["RTX3090"], ["V100"]]},
    }

    # Send requests
    grid = make_grid(possible_values)

    for hyperparams_dict in grid:
        task_settings = ""
        for hp_name, val in hyperparams_dict.items():
            task_settings = f"{task_settings}{hp_name}={val}\n"
        task_settings = task_settings[:-1]
        task_dict = {
            "priority": 1,
            "parameters": { # these are environment variables
                "gin_config": task_settings,
                "COMET_KEY": "OmfvOU0RmHbt4iMa2WIYGBjBf",
            }
        }
        json_dict["tasks"].append(task_dict)
    requests.post("https://jobs.redwoodresearchcompute.com:10101/api", json=json_dict)