import os
import random
import sys
import json
import argparse
from typing import Callable, DefaultDict, List
from collections import defaultdict
import requests
import websocket
import threading
import traceback
import time
from typing import *


token_path = os.path.expanduser("~/.rrjobs-auth-token")

API_URL = "jobs.redwoodresearchcompute.com:10101"
API_SERVER_HOST = f"https://{API_URL}/api"


def get_token() -> str:
    return "tao-1637427484-0-2c83023327cf2d6f68e492599ae49658"
    with open(token_path) as f:
        return f.read().strip()


def api_request(request):
    reqbody = {**request, "token": get_token()}
    print(json.dumps(reqbody))
    return requests.post(API_SERVER_HOST, json=reqbody).json()


def rrjobs_auth(token: str):
    with open(token_path, "w") as f:
        f.write(token + "\n")


def rrjobs_submit(
    name: str,
    git_commit: str,
    command: List[str],
    tasks: list,
    schedulability=True,
    resource_options: List[List[str]] = [["V100"], ["RTX3090"], ["V100"]],
):
    return api_request(
        {
            "kind": "submitJob",
            "name": name,
            "git_commit": git_commit,
            "git_repo": "redwoodresearch/mlab",
            "command": command,
            "tasks": tasks,
            "scheduling_requirements": {
                "schedulability": schedulability,
                "resource_options": resource_options,
            },
        }
    )


ignored_events = [
    "EventLogs",
    "WorldStates",
    "HarnessCheckIns",
    "TaskSchedulingRequirements",
    "JobSchedulingRequirements",
    "TaskRunOrder",
    "GlobalSettings",
    "UserSettings",
    "InstanceTags",
    "EphemeralActivity",
    "CommandAcks",
    "JobStatus",
    "Tasks",
    "Jobs",
]


class RRJobsConnection:
    def __init__(self, on_open=None, on_message=None):
        self.client_on_open = on_open
        self.client_on_message = on_message
        self.response_callbacks = {}
        self.kind_subscriptions = {}
        self.onoutputs = {}
        self.onstatuses = {}
        self.onscrapes = {}
        self.launch_id_to_task_id = {}

        # websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            f"wss://{API_URL}/ws-api",
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        threading.Thread(target=self.ws.run_forever).start()
        print("init")

    def close(self):
        self.ws.close()

    def send(self, message_json, callback=None):
        ack_token = "".join(os.urandom(10).hex())
        message_json["ackToken"] = ack_token
        if callback:
            self.response_callbacks[ack_token] = callback
        self.ws.send(json.dumps(message_json))

    def subscribe_kind(self, kind: str, callback, filters: Optional[dict] = None):
        print("subscribe kind", kind)
        self.kind_subscriptions[kind] = [(callback, filters)]

    def task_output_subscribe(self, task_id: int, callback=None):
        self.send({"kind": "taskOutputSubscribe", "taskId": task_id})
        if callback:
            self.onoutputs[task_id] = [(task_id, callback)]

    def task_output_unsubscribe(self, task_id: int):
        self.send({"kind": "taskOutputUnsubscribe", "taskId": task_id})

    def submit_task(
        self,
        name: str,
        git_commit: str,
        command: List[str],
        tasks: list,
        schedulability=True,
        resource_options: List[List[str]] = [["V100"], ["RTX3090"], ["V100"]],
        on_task_output=None,
        on_task_status=None,
        on_output_scrape=None,
        callback=None,
    ):
        def callback_with_onoutput(task_output):
            task_ids = task_output["data"]["task_ids"]
            for task_id in task_ids:
                if on_task_output:
                    self.task_output_subscribe(task_id, on_task_output)
                if on_task_status:
                    self.onstatuses[task_id] = [on_task_status]
                if on_output_scrape:
                    self.onscrapes[task_id] = [on_output_scrape]
            callback(task_output)

        reqbody = {
            "kind": "submitJob",
            "name": name,
            "git_commit": git_commit,
            "git_repo": "redwoodresearch/mlab",
            "command": command,
            "tasks": tasks,
            "scheduling_requirements": {
                "schedulability": schedulability,
                "resource_options": resource_options,
            },
        }
        print(reqbody)
        self.send(
            reqbody,
            callback_with_onoutput,
        )

    def on_message(self, ws, message):
        message = json.loads(message)
        if self.client_on_message:
            self.client_on_message(message)

        def execute_sub(sub, datum):
            callback, filters = sub
            print(callback)
            callback(datum)
            if not filters or all(
                key in datum and datum[key] == value for key, value in filters.items()
            ):
                pass

        for sub in self.kind_subscriptions.get(message["kind"], ()):
            # print("kind subscription for", message)
            data = message["data"]
            if isinstance(data, list):
                for d in data:
                    execute_sub(sub, d)
            else:
                execute_sub(sub, data)

        if message["kind"] == "auth":
            # FIXME: Check that we authed successfully.
            pass
        elif message["kind"] == "ack":
            if message["token"] in self.response_callbacks:
                self.response_callbacks[message["token"]](message)
                del self.response_callbacks[message["token"]]
        elif message["kind"] == "TaskLaunches":
            for datum in message["data"]:
                self.launch_id_to_task_id[datum["id"]] = datum["task_id"]
        elif message["kind"] == "TaskStatus":
            for datum in message["data"]:
                task_id = datum["task_id"]
                for callback in self.onstatuses.get(task_id, ()):
                    callback(datum)
        elif message["kind"] == "OutputScrapes":
            for datum in message["data"]:
                task_id = datum["task_id"]
                for callback in self.onscrapes.get(task_id, ()):
                    callback(datum)
        elif message["kind"] == "LaunchOutput":
            for datum in message["data"]:
                if datum["launch_id"] in self.launch_id_to_task_id:
                    task_id = self.launch_id_to_task_id[datum["launch_id"]]
                    for task_id, callback in self.onoutputs.get(task_id, ()):
                        callback(task_id, [x[-1] for x in datum["blob"]])
        elif message["kind"] == "streamTaskOutput":
            for datum in message["data"]:
                for task_id, callback in self.onoutputs.get(datum["task_id"], ()):
                    callback(task_id, [x[-1] for x in datum["blob"]])
        elif message["kind"] not in ignored_events:
            print("\x1b[91mUNKNOWN MESSAGE:\x1b[0m", message)

    def on_error(self, ws, error):
        print("\x1b[91mGOT ERROR:", error)
        print(traceback.format_exc(), "\x1b[0m")

    def on_close(self, ws, close_status, close_message):
        print("close")
        if close_status or close_message:
            raise AssertionError(
                f"Orchestrator disconnected unexpectedly: status={close_status} message={close_message}"
            )

    def on_open(self, ws):
        print("websocket opened")

        ws.send(json.dumps({"token": get_token(), "subscriptions": True}))
        self.client_on_open()


if __name__ == "__main__":
    args = parser.parse_args()

    if args.command[0] == "auth":
        if len(args.command) != 2:
            print(
                "Usage: rrjobs auth <token from https://jobs.redwoodresearchcompute.com/settings>",
                file=sys.stderr,
            )
            exit(1)
        rrjobs_auth(args.command[1])
        print("Saved token to", token_path)

    elif args.command[0] == "submit":
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

    elif args.command[0] == "info":
        data = api_request({"kind": "getInfo"}).json()
        import pprint

        pprint.pprint(data)
