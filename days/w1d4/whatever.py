import os
os.system("pip install -r requirements.txt")

from comet_ml import Experiment
from operator import mul
from functools import reduce

import gin
import json
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import w1d4_tests

fname = "days/w1d4/cherry.jpg"
data_train, data_test = w1d4_tests.load_image(fname)

@gin.configurable
class Model(nn.Module):
    def __init__(self, P, H, K):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(P, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, K)

    def forward(self, x0):
        x1 = F.relu(self.linear1(x0))
        x2 = F.relu(self.linear2(x1))
        return self.linear3(x2)
    
@gin.configurable
class Adam():
    def __init__(self, params, lr, betas, eps, weight_decay):
        self.params = list(params)
        self.means = {param: t.zeros_like(param) for param in self.params}
        self.variances = {param: t.zeros_like(param) for param in self.params}
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.time = 0
    
    def zero_grad(self):
        for param in self.params:
            param.grad = None
            
    def step(self):
        with t.no_grad():
            self.time += 1
            for param in self.params:
                g = param.grad + self.weight_decay * param
                self.means[param] = self.betas[0]*self.means[param] + (1 - self.betas[0])*g
                self.variances[param] = self.betas[1]*self.variances[param] + (1 - self.betas[1])*g*g
                mod_means = self.means[param]/(1 - self.betas[0] ** self.time)
                mod_vars = self.variances[param]/(1 - self.betas[1] ** self.time)
                param -= self.lr * mod_means/(t.sqrt(mod_vars) + self.eps)

def evaluate(model, dataloader):
    with t.no_grad():
        loss = 0
        num_inputs = 0
        for input, target in dataloader:
            num_inputs += 1
            output = model(input)
            loss += F.l1_loss(output, target)
        return loss/num_inputs
                
@gin.configurable
def gin_train(model, data_train, data_test, experiment, optimizer_fn, num_epochs, lr, betas, eps, weight_decay):
    epochs = list(range(1, num_epochs + 1))
    for epoch in epochs:
        optimizer = None
        if optimizer_fn == "Adam":
            optimizer = Adam(model.parameters(), lr, betas, eps, weight_decay)
        for input, target in data_train:
            optimizer.zero_grad()
            output = model(input)
            loss = F.l1_loss(output, target)
            loss.backward()
            optimizer.step()
        experiment.log_metric("test_loss", float(evaluate(model, data_test)))

# gin wrangling

def make_grid(possible_values):
    size = reduce(mul, map(len, possible_values.values()))
    ls = []
    for idx in range(size):
        new_values = {}
        for name, values in possible_values.items():
            new_values[name] = values[idx % len(values)]
            idx //= len(values)
        ls.append(new_values)
    return ls

def thangify(pair):
    name, value = pair
    return f"{name}={value}"

def stringify(dictionary):
    return list(map(thangify, list(dictionary.items())))

# possible vals
possible_values = {
    "gin_train.lr": [0.01],
    "Model.P": [2],
    "Model.H": [100],
    "Model.K": [3],
    "gin_train.betas": [(0.9, 0.999)]
}


def hyperparam_search(possible_values):
    grid = make_grid(possible_values)
    for hyperparams in grid:
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([parameters["gin_config"]], stringify(hyperparams))
        model = Model()
        experiment = Experiment(
            api_key="OiNBEOeeT9IFDdHDHRLeEe5hb",
            project_name="image-memorizer",
            workspace="guillecosta",
        )
        experiment.log_parameters(hyperparams)
        gin_train(model, data_train, data_test, experiment)

if __name__ == "__main__":
    parameters_with_config = json.loads(os.getenv("PARAMS"))
    parameters_without_config = json.loads(os.getenv("PARAMS"))
    del parameters_without_config["gin_config"]
    with gin.unlock_config():
        gin.parse_config_files_and_bindings([parameters_with_config["gin_config"]], stringify(parameters_without_config))
    model = Model()
    experiment = Experiment(
        api_key="OiNBEOeeT9IFDdHDHRLeEe5hb",
        project_name="image-memorizer",
        workspace="guillecosta",
    )
    experiment.log_parameters(parameters_without_config)
    gin_train(model, data_train, data_test, experiment)