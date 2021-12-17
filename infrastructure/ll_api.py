import requests
import os
import json
from functools import *

from requests.structures import CaseInsensitiveDict

API_URL = "https://lambdalabs.com/api/cloud"

INSTANCE_TYPES = [
    "gpu.4x.1080ti",
    "gpu.1x.a6000",
    "gpu.2x.a6000",
    "gpu.4x.a6000",
    "gpu.1x.a4000",
    "gpu.8x.v100",
    "gpu.1x.rtx6000",
    "gpu.2x.rtx6000",
    "gpu.4x.rtx6000",
]

ACCOUNT_FILE = os.environ.get("LAMBDA_ACCOUNT_FILE", "./.lambda_account.json")

account = json.loads(open(ACCOUNT_FILE).read())

# copied the api requests made by their webapp into python
# maybe this is useful for this in the future: https://reqbin.com/req/python/c-xgafmluu/convert-curl-to-python-requests


def lambda_req(method, endpoint, data=None):
    func = getattr(requests, method.lower())

    # cookie thing that wont actually do it, may add more cookie jank later
    # session_id = req.cookies.get("sessionid", None)
    # if session_id and session_id != account["session_id"]:
    #     account.session_id = session_id
    #     print("\n\nDIFFERENT SESSION ID RETURNED\n\n")
    #     json.dump(account, open(ACCOUNT_FILE))
    headers = CaseInsensitiveDict()
    headers["authority"] = "lambdalabs.com"
    headers[
        "sec-ch-ua"
    ] = '" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"'
    headers["accept"] = "application/json, text/plain, */*"
    headers["content-type"] = "application/json;charset=UTF-8"
    headers["sec-ch-ua-mobile"] = "?0"
    headers[
        "user-agent"
    ] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    headers["sec-ch-ua-platform"] = '"Windows"'
    headers["origin"] = "https://lambdalabs.com"
    headers["sec-fetch-site"] = "same-origin"
    headers["sec-fetch-mode"] = "cors"
    headers["sec-fetch-dest"] = "empty"
    headers["referer"] = "https://lambdalabs.com/cloud/dashboard/instances"
    headers["accept-language"] = "en-US,en;q=0.9"
    headers[
        "cookie"
    ] = f"intercom-id-ahj4606e=08ac5b9f-9c7e-4292-8274-62693180cd4e; __stripe_mid=72757761-4d6d-47e5-a508-704b3a6c08fc3d1dac; utm_campaign=undefined; utm_content=undefined; _gcl_au=1.1.169599127.1635631193; _fbp=fb.1.1635631193113.177614820; _ga=GA1.2.2140477883.1635631193; wcsid=yLxUZj1iKpcLgAdN184Vi0Wj6k4aWAb6; hblid=7oGXMNuGambC6OFd184Vi0WkAoFbb6aa; _okdetect=%7B%22token%22%3A%2216356312281090%22%2C%22proto%22%3A%22about%3A%22%2C%22host%22%3A%22%22%7D; olfsk=olfsk9282663079934006; _okbk=cd4%3Dtrue%2Cwa1%3Dfalse%2Cvi5%3D0%2Cvi4%3D1635631228640%2Cvi3%3Dactive%2Cvi2%3Dfalse%2Cvi1%3Dfalse%2Ccd8%3Dchat%2Ccd6%3D0%2Ccd5%3Daway%2Ccd3%3Dfalse%2Ccd2%3D0%2Ccd1%3D0%2C; _ok=4536-443-10-4081; _oklv=1635631260747%2CyLxUZj1iKpcLgAdN184Vi0Wj6k4aWAb6; intercom-session-ahj4606e=; sessionid={account['session_id']}; __stripe_sid=13cfe793-0e79-4959-a844-9b896fa87ab753360d"
    kwargies = {
        "headers": headers,
    }
    if data is not None:
        kwargies["data"] = data
        print(data)
    req = requests.post(API_URL + endpoint, **kwargies)
    try:
        return req.json()
    except Exception:
        raise AssertionError("non json reponse\n\n" + req.content.decode("utf-8"))


# this mother fucker doesn't even tell you the instance id!!!!!
def create_instances(instance_type, quantity=1):
    if instance_type not in INSTANCE_TYPES:
        raise AssertionError(
            "allowed instance types are", INSTANCE_TYPES, "not", instance_type
        )
    payload = {
        "method": "launch",
        "params": {
            "ttype": instance_type,  # the first t isn't accidental
            "quantity": quantity,
            "public_key_id": account["public_key"],
            "filesystem_id": None,
        },
    }
    data = lambda_req("post", endpoint="/instances-rpc", data=payload)
    num_created = 0
    if data["error"] is not None:
        for instance_data in data["data"]:
            if instance_data["err"] is not None:
                num_created += 1
    return num_created


def terminate_instances(instance_ids):
    data = lambda_req(
        "post",
        endpoint="/instances-rpc",
        data={"method": "terminate", "params": {"instance_ids": instance_ids}},
    )
    return data


def terminate_all_unprotected():
    instances = list_instance_ids()
    unprotected_instances = [
        instance
        for instance in instances
        if instance not in account["protected_instance_ids"]
    ]
    terminate_instances(unprotected_instances)


def terminate_all_type(type):
    instances = describe_instances()["data"]
    unprotected_instances = [
        instance["id"] for instance in instances if instance["ttype"] == type
    ]
    terminate_instances(unprotected_instances)


def describe_instances():
    req = lambda_req("get", endpoint="/instances")
    if req["error"]:
        raise AssertionError("Lambda Labs SPI error", req)
    return req


def list_instance_ids():
    return [instance["id"] for instance in describe_instances()["data"]]


if __name__ == "__main__":
    # print(describe_instances())
    # terminate_all_type("gpu.4x.a6000")
    print(create_instances("gpu.4x.a6000"))
