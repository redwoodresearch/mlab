from dataclasses import dataclass
from itertools import *
import os
import argparse
from pathlib import Path
import random
import subprocess
import sys
import requests
from typing import *

from utils import import_object_from_qualified_name

parser = argparse.ArgumentParser(description="Submit jobs to Redwood Research")
parser.add_argument("command", nargs="+", help="augh")

token_path = os.path.expanduser("~/.rrjobs-auth-token")

API_URL = "jobs.redwoodresearchcompute.com:10101"
API_SERVER_HOST = f"https://{API_URL}/api"


def get_auth_token() -> str:
    with open(token_path) as f:
        return f.read().strip()


def api_request(request):
    return requests.post(API_SERVER_HOST, json={**request, "token": get_auth_token()}).json()


def rrjobs_auth(token: str):
    with open(token_path, "w") as f:
        f.write(token + "\n")


def rrjobs_submit(
    name: str,
    git_commit: str,
    command: List[str],
    tasks: list,
    schedulability=True,
    resource_options: List[List[str]] = [["A100"], ["RTX3090"], ["V100"]],
):
    return api_request(
        {
            "kind": "submitJob",
            "name": name,
            "git_commit": git_commit,
            "command": command,
            "tasks": tasks,
            "scheduling_requirements": {"schedulability": schedulability, "resource_options": resource_options},
        }
    )


WORKSPACE = Path(__file__).parent.parent
RUNSCRIPT = WORKSPACE / "dev_utils" / "rr_docker"

LAUNCH_TEMPLATE = "redwood-automated-template"


old_settings = None


@dataclass
class Search:
    grid: Dict[str, Iterable[Any]]
    random: Dict[str, Callable[[Dict[str, Any]], Any]]
    gin_config: str
    repeat_count: int = 1
    local: bool = False
    function_path: str = None

    def orchestrate(
        self,
        search_object_location,
        name: str,
        git_commit: str,
        git_branch: str,
    ) -> None:
        # enqueue a bunch of jobs to a queue with a unique name
        # then start a bunch of workers running the work task to pull jobs from the queue

        # sadly we can't do this using a relative path
        # print("checking gin binding works")
        # gin.parse_config_files_and_bindings(config_files=[self.base_config], bindings=[])
        grid = Search.create_grid(self.grid)
        grid_repeated = chain.from_iterable(repeat(grid, self.repeat_count))
        grid_randomized = [Search.create_random(self.random, d) for d in grid_repeated]
        random.shuffle(grid_randomized)
        jobs = [{"bindings": d} for d in grid_randomized]
        # python qa/search.py orchestrate adversarial.search_configs.test_search.mysearch --num_workers=1 --use_rrjobs --git_commit 'ace7428e74e3d9b2b546b4866a5f74e742a2534e'
        current_dir = "/".join(__file__.split("/")[:-1])
        assert current_dir.split("/")[-1] == "qa"
        unity_dir = os.environ["RR_CODE_DIR"]
        gin_dirs = [
            ".",
            unity_dir,
            f"{unity_dir}/adversarial/search_configs",
            f"{unity_dir}/adversarial/search_configs/base_configs",
        ]
        valid_gin_files = [path for path in [d + "/" + self.gin_config for d in gin_dirs] if os.path.exists(path)]
        if len(valid_gin_files) == 0:
            raise AssertionError("can't find gin config", self.gin_config)
        base_config = open(valid_gin_files[0]).read() + "\n"
        if name is None:
            print("using search file name as job name")
            name = search_object_location.split(".")[-2]
        if git_branch:
            git_commit = (
                subprocess.check_output(f'git log -n 1 {git_branch} --pretty=format:"%H"', shell=True)
                .decode("utf-8")
                .strip()
            )
            git_commit_origin = (
                subprocess.check_output(f'git log -n 1 origin/{git_branch} --pretty=format:"%H"', shell=True)
                .decode("utf-8")
                .strip()
            )
            if git_commit != git_commit_origin:
                raise AssertionError(
                    f"WARNING: local and origin git commits do not match. local: {git_commit} origin: {git_commit_origin}"
                )

        elif git_commit is None:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

        # Assert that the git commit is pushed.
        try:
            subprocess.check_output(["git", "branch", "-a", "--contains", git_commit])
        except subprocess.CalledProcessError:
            print(
                f"\x1b[91mGit commit {git_commit} doesn't appear in remote -- did you forget to git push?\x1b[0m",
                file=sys.stderr,
            )
            return

        config_strings = [
            base_config + "\n".join([f"{k} = {repr(v)}" for k, v in task.items()]) for task in grid_randomized
        ]
        print(config_strings[0])
        task_specs = [
            {
                "priority": 1,
                "parameters": {
                    "fn_path": self.function_path,  # FIX!!!!
                    "gin_config": config_string,
                },
            }
            for config_string in config_strings
        ]

        rrjobs_submit(
            name=name,
            git_commit=git_commit,
            git_repo="https://github.com/taoroalin/mlab",
            tasks=task_specs,
            resource_options=["A100", "V100", "RTX3090"],
            command=["python", "adversarial/run_fn_with_config.py"],
        )

    @staticmethod
    def create_grid(grid: Dict[str, Iterable[Any]]) -> list[Dict[str, Any]]:
        # turn each list into a list of tuples of (k, value), then call dict on the product of those lists, so that
        # each k gets associated to each value in all possible combinations
        return list(map(lambda d: dict(d), product(*map(lambda kxs: map(lambda x: (kxs[0], x), kxs[1]), grid.items()))))

    @staticmethod
    def create_random(
        random: Dict[str, Callable[[Dict[str, Any]], Any]], initial_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        cur_dict = initial_dict
        for k, func in random.items():
            cur_dict[k] = func(cur_dict)
        return cur_dict


if __name__ == "__main__":
    # given a function to run, parse a number of command line arguments and then delegate to either work or orchestrate
    parser = argparse.ArgumentParser(
        description="Run a search and then run a worker which works to execute the search code"
    )
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "search_object_location",
        help="a location starting in the top level unity directory specifying a search object"
        "the last (period separated) section of the location should be a top level name in the "
        "specified module",
    )

    parser.add_argument("--local", action="store_true", help="if given, run the tasks sequentially on this machine")
    parser.add_argument("--name", type=str, help="name of job in rrjobs", default=None)
    parser.add_argument("--git_commit", type=str, help="name of git commit, defaults to current commit", default=None)
    parser.add_argument("--git_branch", type=str, help="name of git branch, MUST BE PUSHED!", default=None)

    # parser_orchestrate.add_argument(
    #     "local_gpus",
    #     help="whether to run on multiple local processes with different gpus instead of different instances",
    # )

    args = parser.parse_args()

    search_object: Search = import_object_from_qualified_name(args.search_object_location)

    if args.task == "orchestrate":
        search_object.orchestrate(
            search_object_location=args.search_object_location,
            local=args.local,
            name=args.name,
            git_commit=args.git_commit,
            git_branch=args.git_branch,
        )

if __name__ == "__main__":
    args = parser.parse_args()

    print(
        rrjobs_submit(
            name="snp-count-up",
            git_commit="1a34c60e7ef36415beafbf46aec2da187b9cc1d7",
            command=["python", "rrjobs/backend/example_job.py"],
            tasks=[
                {"priority": 0, "parameters": {"count_to": 300}},
                {"priority": 1, "parameters": {"count_to": 300}},
                {"priority": 2, "parameters": {"count_to": 300}},
                {"priority": 3, "parameters": {"count_to": 300}},
            ],
            resource_options=[["V100"]],
        )
    )
