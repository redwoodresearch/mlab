from comet_ml import Experiment

# Run your code and go to /

import torch
t = torch
from torch import nn, optim
import w1d4_tests
import matplotlib.pyplot as plt
import w1d4_sol as sol
import gin

import os
os.system("pip install -r ../../requirements.txt")

from .w1d4_template import run_experiment
data_train, data_test = w1d4_tests.load_image('pineapple.jpg')

@gin.configurable
def train(model: nn.Module, dataloader, lr, momentum, experiment=None, epochs=1):
    for e in range(epochs):
        sol._train(model, dataloader, lr, momentum)
        if experiment is not None:
            experiment.log_metric('val loss', sol._evaluate(model, dataloader))

def make_grid(possible_values):
    possible_values = possible_values.items()
    result = [{}]
    for k, vals in possible_values:
        result = [dict(**x, **{k: v}) for x in result for v in vals]
    return result

# print(make_grid(possible_values))

def run_experiment(params):
    with gin.unlock_config():
        gin.parse_config_files_and_bindings([], [f'{k}={v}' for k,v in params.items()])
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="oGcm04SiEeJM89dRyU0vcOFzd",
            project_name="hyperparameters4",
            workspace="luuf",
        )
        for k, v in params.items():
            assert isinstance(k, str)
            print(k)
            print(v)
            experiment.log_parameter(k, v)
        m = sol._MLP(in_dim=2, hidden_dim=400, out_dim=3)
        train(m, data_train)
        experiment.end()


run_experiment(os.environ['PARAMS'].gin_config)
