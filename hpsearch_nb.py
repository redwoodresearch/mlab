import argparse
from typing import *
import gin
import os
import json
from days.utils import import_object_from_qualified_name
import sys


def run_fn_with_config(fnpath: str, config: str):
    # given a function and a gin config, run it with the config

    temp_name = "TEMP_CONFIG.gin"

    fn = import_object_from_qualified_name(fnpath)

    with open(temp_name, "w+") as text_file:
        text_file.write(config)
    gin_search_path = f"{os.getcwd()}"
    gin.add_config_file_search_path(gin_search_path)
    gin.parse_config_files_and_bindings(config_files=[temp_name], bindings=[])
    fn()


if __name__ == "__main__":
    params = json.loads(os.environ["PARAMS"])
    print("params", params)
    run_fn_with_config(params["fn_path"], params["gin_config"])
