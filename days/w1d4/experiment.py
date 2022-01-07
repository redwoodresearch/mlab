from comet_ml import Experiment

ben_experiment_params = {
    'api_key': "absncaDYNLt6jpNh1Ez0OIVTe",
    'project_name': "mlab",
    'workspace': "bmillwood",
}

arthur_experiment_params = {
    'api_key': "cWQ5thtmlU2pZH62GZVFghchU",
    'project_name': "general",
    'workspace': "arthurconmy",
}

import os
os.system("pip install -r requirements.txt")
import sys

if "--arthur" in sys.argv:
    experiment_params = arthur_experiment_params
else:
    experiment_params = ben_experiment_params

import torch as t
import numpy as np
import w1d4_tests
import matplotlib.pyplot as plt
import gin
import json

fname = "days/w1d4/raichu.png"

data_train, data_test =  w1d4_tests.load_image(fname)

from PIL import Image
img = Image.open(fname)
from torchvision import transforms
tensorize = transforms.ToTensor()
img = tensorize(img)

@gin.configurable
class RaichuModel(t.nn.Module):
    def __init__(self, P, H, K):
        super().__init__()
        self.layers = t.nn.Sequential(t.nn.Linear(P, H), 
            t.nn.ReLU(),
            t.nn.Linear(H, H),
            t.nn.ReLU(),
            t.nn.Linear(H, K),
        )

    def forward(self, x):
        return self.layers(x)

@gin.configurable
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.first_moment = [0.0 for param in self.params]
        self.second_moment = [0.0 for param in self.params]
        self.t = 0

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        with t.no_grad():
            self.t += 1
            for i, param in enumerate(self.params):
                g = param.grad
                g += self.weight_decay * param
                b1, b2 = self.betas
                self.first_moment[i] *= b1
                self.first_moment[i] += (1 - b1)*g
                self.second_moment[i] *= b2
                self.second_moment[i] += (1 - b2)*g**2
                m_hat = self.first_moment[i] / (1 - b1**self.t)
                v_hat = self.second_moment[i] / (1 - b2**self.t)
                param -= self.lr * m_hat/(v_hat**0.5 + self.eps)

# w1d4_tests.test_adam(Adam)

@gin.configurable
def train(model, dataloader, optimizer_kind='SGD'): # -> trained model
    if optimizer_kind == 'SGD':
        optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_kind == 'RMSprop':
        optimizer = t.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_kind == 'Adam':
        optimizer = Adam(model.parameters())

    for X, Y in dataloader:
        optimizer.zero_grad()
        output = model(X)
        loss = t.nn.functional.l1_loss(output, Y)
        loss.backward()
        optimizer.step()

    return model

def evaluate(model, dataloader):
    cumulative_loss = 0
    for X, Y in dataloader:
        loss = t.nn.functional.l1_loss(model(X), Y)
        cumulative_loss += loss.detach()
    return cumulative_loss / len(dataloader)

@gin.configurable
def trains(model, data_train, data_test, num_epochs):
    for _ in range(num_epochs):
        train(model=model, dataloader=data_train)
    return evaluate(model, data_train)

with gin.unlock_config():
    json_loaded = json.loads(os.environ['PARAMS'])
    print(json_loaded)
    config = json_loaded["gin_config"]
    gin.parse_config_files_and_bindings(["config.gin"], bindings=config)
    experiment = Experiment(**experiment_params)
    log_lr = np.log10(gin.get_bindings(Adam)['lr'])
    experiment.log_parameter('log_lr', log_lr)
    hidden_size = gin.get_bindings(RaichuModel)['H']
    experiment.log_parameter('hidden_size', hidden_size)
    model = RaichuModel(P=2, K=3)
    test_loss = trains(model, data_train, data_test)
    experiment.log_metric('test_loss', test_loss)
    experiment.end()
