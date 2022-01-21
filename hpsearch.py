import argparse
import re
import subprocess
import json
import os
import sys
import shlex
import atexit
import uuid
import math
import traceback
import sys
import rrjobs  # type: ignore
import itertools
from itertools import chain, product, repeat
from typing import Any, Callable, Iterable
from dataclasses import dataclass
import shutil
import termios
import random
import shlex
from typing import *

import traceback
import requests as requests

import gin

from pathlib import Path

from days.utils import (
    import_object_from_qualified_name,
)

from typing import Optional, Union

WORKSPACE = Path(__file__).parent.parent
RUNSCRIPT = WORKSPACE / "dev_utils" / "rr_docker"

LAUNCH_TEMPLATE = "redwood-automated-template"

old_settings = None


def init_anykey():
    global old_settings
    old_settings = termios.tcgetattr(sys.stdin)
    new_settings = termios.tcgetattr(sys.stdin)
    new_settings[3] = new_settings[3] & ~(termios.ECHO | termios.ICANON)  # lflags
    new_settings[6][termios.VMIN] = 0  # cc
    new_settings[6][termios.VTIME] = 1  # cc
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, new_settings)


@atexit.register
def term_anykey():
    global old_settings
    if old_settings:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def anykey():
    while True:
        c = os.read(sys.stdin.fileno(), 1)
        if c:
            return c


class LineCollector:
    def __init__(self, max_lines: int):
        self.max_lines = max_lines
        self.all_lines: List[str] = []
        self.contents = ["~"] * max_lines

    def update(self, new_output: List[str]):
        for entry in new_output:
            for line in entry.split("\n"):
                self.all_lines.append(line)
                self.contents.append(line)
        # Keep just the last max_lines lines.
        self.contents = self.contents[len(self.contents) - self.max_lines :]


class Interface:
    def __init__(self):
        self.job_id = None
        self.task_ids = None
        self.is_running = True
        self.term_width = 80
        self.term_height = 20
        self.update_size()
        # We use this back_up variable to track how many lines we've drawn during the repaint,
        # so we can back up over the right number for the next repaint, to redraw over it.
        self.back_up = 0
        # Maps a task_id to a TaskStatus JSON object.
        self.status_rows: Dict[int, Any] = {}
        # Maps a task_id to the blob field of an OutputScrapes JSON object.
        self.scrapes: Dict[int, Any] = {}
        # Maps a task_id to a LineCollector with the last few lines of stdout/stderr.
        self.tasks_last_few_lines: Dict[int, LineCollector] = {}
        self.logs_mode = False

    def update_size(self):
        """update_size(self) -> NoneType

        Determine the width
        """
        size = shutil.get_terminal_size((80, 20))
        self.term_width = size.columns
        self.term_height = size.lines

    def width_limiting_print(self, s: str):
        """width_limiting_print(self, s: str) -> NoneType

        Print a string but without
        """
        assert "\n" not in s
        # FIXME: This is all totally busted, and can still cut things off part way.
        # Our goal is to figure out how many cells this string will take up in the terminal, so we need to remove VT-100 escapes.
        actual_length = len(re.sub("\x1b\\[[^m]*m", "", s))
        to_chop_off = max(0, actual_length - self.term_width)
        print(s[: len(s) - to_chop_off])
        self.back_up += 1

    def erase(self):
        self.update_size()
        # This VT-100 escape sequence backs up and erases a line, thus letting us repaint the screen.
        sys.stdout.write("\x1b[1A\x1b[2K" * self.back_up)
        # Begin a repaint!
        self.back_up = 0

    def repaint(self):
        self.erase()
        # Use VT-100 color escape sequences to color the various statuses.
        colorize = {
            "pending": "\x1b[95mpending\x1b[0m  ",
            "queued": "\x1b[94mqueued\x1b[0m   ",
            "running": "\x1b[96mrunning\x1b[0m  ",
            "done": "\x1b[92mdone\x1b[0m     ",
            "crashed": "\x1b[91mcrashed\x1b[0m  ",
            "killed": "\x1b[93mkilled\x1b[0m   ",
            "cancelled": "\x1b[93mcancelled\x1b[0m",
        }
        # Write down all the tasks in reverse order, so the task most likely to be scheduled first is at the bottom.
        # This is just to make sure you can see the task getting scheduled even if the task list is too long to fit on your screen.
        for task_id in self.task_ids[::-1]:
            if (
                task_id not in self.status_rows
                or task_id not in self.scrapes
                or task_id not in self.tasks_last_few_lines
            ):
                continue
            status_row = self.status_rows[task_id]
            return_code = status_row["return_code"]
            if return_code is None:
                return_code = "..."
            comet_url = self.scrapes[task_id]["comet_url"]
            comet = " " + comet_url if comet_url else ""
            self.width_limiting_print(
                f"\x1b[94m========== Task\x1b[0m {task_id:5}  Status: {colorize.get(status_row['status'])}  Return code: {return_code:3}{comet}"
            )
            # Print the last few lines of stdout/stderr for this task.
            for line in self.tasks_last_few_lines[task_id].contents:
                self.width_limiting_print(line)
        self.width_limiting_print(
            f"\x1b[94m=====\x1b[0m https://jobs.redwoodresearchcompute.com/jobs/j{self.job_id}"
        )
        self.width_limiting_print(
            "\x1b[94m=====\x1b[0m k cancel job | K cancel+archive job | q close, keep running | l show crash logs"
        )

    def on_submitted(self, row):
        self.job_id = row["data"]["job_id"]
        self.task_ids = row["data"]["task_ids"]
        self.status_rows = {
            task_id: {"status": "???", "return_code": None} for task_id in self.task_ids
        }
        # line_count is the number of lines of stdout/stderr we show for each task. I split up the height for each task evenly, and round down.
        self.line_count = max(
            0,
            math.floor(
                (self.term_height - 5 - len(self.task_ids)) / max(1, len(self.task_ids))
            ),
        )
        # However, make sure that the very first task in the list always gets at least 5 lines of stdout/stderr shown.
        self.first_lines = max(5, self.line_count)
        self.tasks_last_few_lines = {
            task_id: LineCollector(
                self.first_lines if task_id == self.task_ids[0] else self.line_count
            )
            for task_id in self.task_ids
        }
        self.scrapes = {task_id: {"comet_url": None} for task_id in self.task_ids}
        if not self.logs_mode:
            self.repaint()

    def on_task_status(self, row):
        if not self.is_running:
            return
        self.status_rows[row["task_id"]] = row
        if not self.logs_mode:
            self.repaint()

    def on_task_output(self, task_id: int, output_chunks: List[str]):
        if not self.is_running:
            return
        self.tasks_last_few_lines[task_id].update(output_chunks)
        if not self.logs_mode:
            self.repaint()

    def on_output_scrape(self, row):
        self.scrapes[row["task_id"]] = row["blob"]

    def main(self):
        while True:
            k = anykey()
            if k == b"q":
                self.width_limiting_print(
                    f"\x1b[94m==========\x1b[0m Exiting, letting the job continue running at: https://jobs.redwoodresearchcompute.com/jobs/j{self.job_id}"
                )
                return
            if k == b"k":
                self.width_limiting_print(
                    "\x1b[91m==========\x1b[0m Exiting, cancelling the job"
                )
                self.is_running = False
                return "cancel"
            if k == b"K":
                self.width_limiting_print(
                    "\x1b[91m==========\x1b[0m Exiting, cancelling + archiving the job"
                )
                self.is_running = False
                return "archive"
            if k == b"l":
                self.logs_mode = not self.logs_mode
                if self.logs_mode:
                    self.erase()
                    for task_id, row in self.status_rows.items():
                        if row["status"] == "crashed":
                            self.width_limiting_print(
                                "\x1b[91m⇓⇓⇓⇓⇓⇓⇓⇓⇓⇓ BEGIN LOGS ⇓⇓⇓⇓⇓⇓⇓⇓⇓⇓\x1b[0m"
                            )
                            for line in self.tasks_last_few_lines[task_id].all_lines:
                                self.width_limiting_print(line)
                            self.width_limiting_print(
                                "\x1b[91m⇑⇑⇑⇑⇑⇑⇑⇑⇑⇑ END LOGS ⇑⇑⇑⇑⇑⇑⇑⇑⇑⇑\x1b[0m"
                            )
                            self.width_limiting_print(
                                f"\x1b[91m=====\x1b[0m Task {task_id} crashed with retcode {row['return_code']}"
                            )
                            self.width_limiting_print(
                                "\x1b[94m=====\x1b[0m k cancel job | K cancel+archive job | q close, keep running | l hide crash logs"
                            )
                            break
                    else:
                        self.width_limiting_print(
                            "\x1b[91m===== No job with a crash -- hit l again to exit\x1b[0m"
                        )
                else:
                    self.repaint()


@dataclass
class Job:
    parameters: Dict[str, Any]
    gin_config: str


def make_grid(axes):
    return [
        {key: value for key, value in zip(axes.keys(), values_choice)}
        for values_choice in itertools.product(*axes.values())
    ]


def hpsearch(name, fn_path, base_config, search_spec, comet_key, local=True):
    base_config = open(base_config).read()
    init_anykey()
    interface = Interface()
    git_commit = (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )
    remote_branches_str = subprocess.check_output(
        ["git", "branch", "-r", "--contains", git_commit]
    )
    if not remote_branches_str:
        print(
            f"\x1b[91mGit commit {git_commit} doesn't appear in remote -- did you forget to git push?\x1b[0m",
            file=sys.stderr,
        )
        return
    grid = make_grid(search_spec)
    if len(grid) > 40:
        raise AssertionError(
            f"During MLAB, let's only run 40 different hyperparameter combinations at once. You submitted {len(grid)}."
        )

    random.shuffle(grid)
    config_strings = [
        base_config + "\n".join([f"{k} = {repr(v)}" for k, v in task.items()])
        for task in grid
    ]
    print(config_strings[0])
    task_specs = [
        {
            "priority": 1,
            "parameters": {
                "fn_path": fn_path,  # FIX!!!!
                "gin_config": config_string,
                "name": name,
                "comet_key": comet_key,
            },
        }
        for config_string in config_strings
    ]

    def on_open():
        print("search onopen")
        rrjobs_connection.submit_task(
            name=name,
            git_commit=git_commit,
            tasks=task_specs,
            command=["python", "run_fn_with_config.py"],
            on_task_output=interface.on_task_output,
            on_task_status=interface.on_task_status,
            on_output_scrape=interface.on_output_scrape,
            callback=interface.on_submitted,
        )

    if local:
        print("LOCAL TRIAL RUN")
        for task_spec in task_specs:
            my_env = {"PARAMS": json.dumps(task_spec["parameters"]), **os.environ}
            proc = subprocess.Popen(["python", "run_fn_with_config.py"], env=my_env)
            outs, errs = proc.communicate()
            assert proc.returncode == 0
        return

    rrjobs_connection = rrjobs.RRJobsConnection(on_open=on_open)

    action = interface.main()
    if action in ("cancel", "archive") and interface.task_ids:
        rrjobs_connection.send({"kind": "killTasks", "tasks": interface.task_ids})
    if action == "archive":
        rrjobs_connection.send({"kind": "archiveJobs", "jobs": [interface.job_id]})
    rrjobs_connection.close()
