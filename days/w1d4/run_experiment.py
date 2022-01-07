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
import json
os.system("pip install -r ../../requirements.txt")

data_train, data_test = w1d4_tests.load_image('days/w1d4/pineapple.jpg')

@gin.configurable
def train(model: nn.Module, dataloader, lr, momentum, experiment=None, epochs=1):
    for e in range(epochs):
        sol._train(model, dataloader, lr, momentum)
        if experiment is not None:
            experiment.log_metric('val loss', sol._evaluate(model, dataloader))


# print(make_grid(possible_values))

def run_experiment(params):
    with gin.unlock_config():
        gin.parse_config_files_and_bindings([], params.split('\n'))
        # Create an experiment with your api key
        experiment = Experiment(
            api_key="oGcm04SiEeJM89dRyU0vcOFzd",
            project_name="hyperparameters5",
            workspace="luuf",
        )
        experiment.log_parameter('hyperparameters', params)
        m = sol._MLP(in_dim=2, hidden_dim=400, out_dim=3)
        train(m, data_train)


print(os.environ['PARAMS'])
run_experiment(json.loads(os.environ['PARAMS'])['gin_config'])
# run_experiment('train.lr=.01\ntrain.epochs=1\ntrain.momentum=.9')
