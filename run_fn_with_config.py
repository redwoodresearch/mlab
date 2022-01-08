import argparse
from typing import *
import gin
import os
import json
import adversarial.utils
from adversarial.utils import import_object_from_qualified_name
import sys


def run_fn_with_config(fnpath: str, config: str):
    # given a function and a gin config, run it with the config

    temp_name = "adversarial/search_configs/TEMP_CONFIG.gin"
    try:
        os.system(
            "apt-get install software-properties-common; curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash; apt-get install git-lfs; git init; git lfs install"
        )
    except Exception as e:
        print(e)
    if "RR_CODE_DIR" not in os.environ:
        os.environ["RR_CODE_DIR"] = os.getcwd()
    if "DATABASE_URL" not in os.environ:
        os.environ[
            "DATABASE_URL"
        ] = "postgresql://ucs351014et62t:pe9a00d8709b35f82b2ee868a1d13843d65386da5d06eff7a238e41c6a0268ca2@ec2-18-214-1-96.compute-1.amazonaws.com:5432/d1fsrok1f8tlps"

    fn = import_object_from_qualified_name(fnpath)

    with open(temp_name, "w+") as text_file:
        text_file.write(config)
    gin_search_path = f"{os.environ['RR_CODE_DIR']}/adversarial/search_configs"
    gin.add_config_file_search_path(gin_search_path)
    gin.add_config_file_search_path(gin_search_path + "/base_configs")
    gin.parse_config_files_and_bindings(config_files=[temp_name], bindings=[])
    fn()


if __name__ == "__main__":
    params = json.loads(os.environ["PARAMS"])
    print("params", params)
    run_fn_with_config(params["fn_path"], params["gin_config"])
