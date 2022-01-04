from utils import import_object_from_qualified_name
import requests
import os
from typing import *
import subprocess


API_URL = f"https://jobs.redwoodresearchcompute.com:10101/api"


def get_git_commit():
    return (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )


def rrjobs_submit(
    name: str,
    command: List[str],
    tasks: list,
    resource_options: List[List[str]] = [["A100"], ["RTX3090"], ["V100"]],
):
    req = requests.post(
        API_URL,
        json={
            "token": "tao-1637427484-0-2c83023327cf2d6f68e492599ae49658",
            "kind": "submitJob",
            "git_repo": "redwoodresearch/mlab",
            "git_commit": get_git_commit(),
            "name": name,
            "command": command,
            "tasks": tasks,
            "scheduling_requirements": {
                "schedulability": True,
                "resource_options": resource_options,
            },
        },
    )
    return req.json()


if __name__ == "__main__":
    print(
        rrjobs_submit(
            "mlab_test_job",
            ["echo", "hi"],
            [{"priority": 1, "parameters": {"gin_config": ""}}],
        )
    )
