import os

os.system("pip install -r requirements.txt")
os.system("pip install -e .")

from comet_ml import Experiment
import torch
import numpy as np
from typing import *
import gin
import os
import json
from utils import import_object_from_qualified_name


def log_all_gin_parameters(experiment):
    """
    This function is largely stolen from gin.config.config_str(), by the way
    """
    for (scope, selector), config in gin.config._CONFIG.items():
        configurable_ = gin.config._REGISTRY[selector]
        if configurable_.wrapped in (gin.config.macro, gin.config._retrieve_constant):
            continue
        minimal_selector = gin.config._minimal_selector(configurable_)
        scoped_selector = (scope + "/" if scope else "") + minimal_selector
        for arg, val in sorted(config.items()):
            experiment.log_parameter(f"{scoped_selector}.{arg}", str(val))


@gin.configurable
def set_random_seed(seed=0):
    torch.manual_seed(seed)
    # np.random.seed(seed)


def run_fn_with_config(fnpath: str, config: str, name: str, comet_key: str):
    """given a function and a gin config, run it with the config"""

    temp_name = "TEMP_CONFIG.gin"

    fn = import_object_from_qualified_name(fnpath)

    with open(temp_name, "w+") as text_file:
        text_file.write(config)
    gin_search_path = f"{os.getcwd()}"
    gin.add_config_file_search_path(gin_search_path)
    gin.parse_config_files_and_bindings(config_files=[temp_name], bindings=[])

    experiment = Experiment(project_name=name, api_key=comet_key)
    log_all_gin_parameters(experiment)
    set_random_seed()
    fn_return = fn(experiment=experiment)
    if fn_return is not None:
        fname = "tempmodel.pt"
        torch.save(fn_return, fname)
        experiment.log_model("model", fname)


if __name__ == "__main__":
    params = json.loads(os.environ["PARAMS"])
    print("params", params)
    run_fn_with_config(
        params["fn_path"], params["gin_config"], params["name"], params["comet_key"]
    )
