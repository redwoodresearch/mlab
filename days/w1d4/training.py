# Import comet_ml at the top of your file
from comet_ml import Experiment

import torch
from torch import nn
from torch import optim
import w1d4_tests
import matplotlib.pyplot as plt
import subprocess
import gin
import os


def get_git_commit():
    return (
        subprocess.check_output("git rev-parse HEAD", shell=True)
        .decode("utf-8")
        .strip()
    )


def get_fullpath(f):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f)

fname = get_fullpath("picture.jpeg")
data_train, data_test =  w1d4_tests.load_image(fname)

class MLP(nn.Module):
    def __init__(self, P, H, K) -> None:
        super().__init__()

        self.layer1 = nn.Linear(P, H)
        self.layer2 = nn.Linear(H, H)
        self.layer3 = nn.Linear(H, K)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)        
        return x

def _evaluate(model, dataloader):
    loss_fn = nn.L1Loss()
    loss = 0
    entries = 0
    for input, target in dataloader:
        output = model(input)
        loss += loss_fn(output, target)
        entries += 1
    return loss / entries
evaluate = _evaluate

class SGD:
    def __init__(self, params, lr, momentum, dampening, weight_decay):
        self.params = list(params)
        self.bt = [None] * len(self.params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay

        self.t = 1

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        # take gradient
        for i, param in enumerate(self.params):

            gt = param.grad
            if self.weight_decay != 0:
                gt = gt + self.weight_decay * param.data
            if self.momentum != 0:
                if self.t > 1:
                    bt = self.momentum * self.bt[i] + (1 - self.dampening) * gt
                else:
                    bt = gt
                
                self.bt[i] = gt = bt
            param.data = param.data - self.lr * gt
        self.t += 1

class RMSProp:
    def __init__(self, params, lr, alpha, eps, weight_decay, momentum):
        self.params = list(params)
        self.bt = [0] * len(self.params)
        self.vt = [0] * len(self.params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.weight_decay = weight_decay

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        # take gradient
        for i, param in enumerate(self.params):

            gt = param.grad
            if self.weight_decay != 0:
                gt = gt + self.weight_decay * param.data

            vt = self.alpha * self.vt[i] + (1 - self.alpha) * gt**2
            
            if self.momentum > 0:
                bt = self.momentum * self.bt[i] + gt / (torch.sqrt(vt) + self.eps)
                self.bt[i] = bt
                param.data = param.data - self.lr * bt
            else:
                param.data = param.data - self.lr * gt / (torch.sqrt(vt) + self.eps)
            self.vt[i] = vt

class Adam:
    def __init__(self, params, lr, betas, eps, weight_decay):
        self.params = list(params)
        self.m = [0] * len(self.params)
        self.v = [0] * len(self.params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 1

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        # take gradient
        for i, param in enumerate(self.params):
            gt = param.grad
            if self.weight_decay != 0:
                gt = gt + self.weight_decay * param.data
            mt = self.betas[0] * self.m[i] + (1 - self.betas[0]) * gt
            vt = self.betas[1] * self.v[i] + (1 - self.betas[1]) * gt**2
            mt_hat = mt / (1 - self.betas[0] ** self.t)
            vt_hat = vt / (1 - self.betas[1] ** self.t)
            param.data = param.data - self.lr * mt_hat / (torch.sqrt(vt_hat) + self.eps)

            self.m[i] = mt
            self.v[i] = vt

        self.t += 1

def train_with_optimizer(optimizer, m, dataloader):
    # optimizer represents the algorithm we use for updating parameters
    loss_fn = nn.L1Loss()
    for input, target in dataloader:
        optimizer.zero_grad()	# sets param.grad to None for linked params
                        # (this prevents accumulation of gradients)
        output = m(input)		  # the model’s predictions
        loss = loss_fn(output, target)	  # measure how bad predictions are
        loss.backward()			  # calculate gradients
        optimizer.step()	
    return m	        


OPTIMIZERS = {
    "adam": [Adam, "lr", "betas", "eps", "weight_decay"],
    "sgd": [SGD, "lr", "momentum", "dampening", "weight_decay"],
    "rms": [RMSProp, "lr", "alpha", "eps", "weight_decay", "momentum"]
}


@gin.configurable
def model(hidden_size):
    print("tf", hidden_size)
    return MLP(2, hidden_size, 3)

@gin.configurable
def evaluate():
    pass

@gin.configurable
def train(epochs, optimizer, learning_rate=None, lr=None, betas=None, eps=None, weight_decay=None, momentum=None, dampening=None):
    opt_data = OPTIMIZERS[optimizer]
    print(locals())
    l = locals()
    opt_args = {
        i: l[i] for i in opt_data[1:]
    }

    m = model()
    optim = opt_data[0](m.parameters(), **opt_args)

    for epoch in range(epochs):
        m = train_with_optimizer(optim, m, data_train)


import numpy as np
from itertools import product
def make_grid(possible_values):
    all_combinations = product(*[[(key, value) for value in values] for key, values in possible_values.items() ])
    return list(map(dict, all_combinations))


experiment = Experiment(
    api_key="KSFWCUmnhYqYNp5i825EEdiQk",
    project_name=f"hyperparameter_search",
    workspace="alwin-peng",
)

gin_config = os.getenv("PARAMS", """train.epochs = 50
train.optimizer = "sgd"
train.lr = 0.001
train.weight_decay = 0.05
train.momentum = 0.9
train.dampening = 0
model.hidden_size = 400""")

import json
params = json.loads(os.environ["PARAMS"])



with gin.unlock_config():
    gin.parse_config_files_and_bindings([get_fullpath("config.gin")], params["GIN_CONFIG"])
    experiment.log_parameters({"params": gin_config})
    train()  
    experiment.end()
