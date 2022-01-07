from comet_ml import Experiment
from tqdm import tqdm

arthur_experiment_params = {
    'api_key': "cWQ5thtmlU2pZH62GZVFghchU",
    'project_name': "general",
    'workspace': "arthurconmy",
}
import sys

if "--arthur" in sys.argv:
    experiment_params = arthur_experiment_params

import torch as t
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import einops
import einops
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
from typing import Tuple 
# import w1d4_tests
import matplotlib.pyplot as plt
import gin

def load_image(fname, n_train=8192, batch_size=128):
    img = Image.open(fname)
    tensorize = transforms.ToTensor()
    img = tensorize(img)
    img = einops.rearrange(img, "c h w -> h w c")
    height, width = img.shape[:2]

    n_trn = n_train
    n_tst = 1024
    X1 = t.randint(0, height, (n_trn + n_tst,))
    X2 = t.randint(0, width, (n_trn + n_tst,))
    X = t.stack([X1.float() / height - 0.5, X2.float() / width - 0.5]).T
    Y = img[X1, X2] - 0.5

    Xtrn, Xtst = X[:n_trn], X[n_trn:]
    Ytrn, Ytst = Y[:n_trn], Y[n_trn:]

    dl_trn = DataLoader(TensorDataset(Xtrn, Ytrn), batch_size=batch_size, shuffle=True)
    dl_tst = DataLoader(TensorDataset(Xtst, Ytst), batch_size=batch_size)
    return dl_trn, dl_tst

fname = "/home/arthur/Documents/ML/MLAB/mlab/days/w1d4/raichu.png"

data_train, data_test = load_image(fname)

from PIL import Image
img = Image.open(fname)
from torchvision import transforms
tensorize = transforms.ToTensor()
img = tensorize(img)

@gin.configurable
class RaichuModel(t.nn.Module):
    def __init__(self, P, hidden_layer_sizes, K):
        super().__init__()

        sequential_list = [t.nn.Linear(P, hidden_layer_sizes[0]), t.nn.ReLU()]
        for i in range(1, len(hidden_layer_sizes)):
            sequential_list.append(t.nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
            sequential_list.append(t.nn.ReLU())
        sequential_list.append(t.nn.Linear(hidden_layer_sizes[-1], K))

        self.layers = t.nn.Sequential(*sequential_list)

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

from time import time

@gin.configurable
def trains(model, data_train, data_test, num_epochs):
    start_time = time()
    while time() - start_time < 60:
        train(model=model, dataloader=data_train)
    return evaluate(model, data_train)

config_space = []

for layers in range(1):
    
    divs = []
    for d in range(2, 25):
        if 100 % d == 0:
            divs.append([100 // d for _ in range(d)])
    print(divs)

    for Hs in divs:
        config = [
            f"RaichuModel.hidden_layer_sizes = {Hs}",
        ] 
        config_space.append(config)

for config in config_space:
    with gin.unlock_config():
        print(f"Starting config {config}")
        gin.parse_config_files_and_bindings(["config.gin"], bindings=config)
        experiment = Experiment(**experiment_params)
        # log_lr = np.log10(gin.get_bindings(Adam)['lr'])
        # experiment.log_parameter('log_lr', log_lr)
        # hidden_size = gin.get_bindings(RaichuModel)['H']
        # experiment.log_parameter('hidden_size', hidden_size)
        hidden_layer_sizes = gin.get_bindings(RaichuModel)['hidden_layer_sizes']
        experiment.log_parameter('hidden_sizes', hidden_layer_sizes)
        model = RaichuModel(P=2, K=3)
        test_loss = trains(model, data_train, data_test)
        print(f"TESTLOSS: {test_loss}")
        experiment.log_metric('test_loss', test_loss)
        experiment.end()
input("ENNDD")

from time import ctime
start_time = ctime()

def show_my_image(img, model):
    img_depth, img_height, img_width = img.shape
    result = t.zeros(img.shape)

    height_indices = t.arange(img_height, dtype=t.float)
    height_indices /= img_height
    height_indices -= 0.5

    width_indices = t.arange(img_width, dtype=t.float)
    width_indices /= img_width
    width_indices -= 0.5

    indices = t.stack((
        height_indices.unsqueeze(1).expand(img_height, img_width),
        width_indices.unsqueeze(0).expand(img_height, img_width)
        ), 2
    )
    result = model(indices).detach()
    result += 0.5
    result = result.unsqueeze(-1)
    plt.imsave("file5.png", result)

gin.parse_config_files_and_bindings(["config.gin"], bindings=config)
model=RaichuModel(P=2, K=3)
trains(model, data_train, data_test)