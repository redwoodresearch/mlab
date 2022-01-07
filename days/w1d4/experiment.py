from comet_ml import Experiment

experiment_params = {
    'api_key': "absncaDYNLt6jpNh1Ez0OIVTe",
    'project_name': "mlab",
    'workspace': "bmillwood",
}

arthur_experiment_params = Experiment(
    api_key="cWQ5thtmlU2pZH62GZVFghchU",
    project_name="general",
    workspace="arthurconmy",
)

experiment = Experiment(**experiment_params)

import torch as t
import numpy as np
import w1d4_tests
import matplotlib.pyplot as plt
import gin

fname = "/home/ubuntu/mlab/days/w1d4/raichu.png"

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

def log_gin_parameter(configurable, name):
    experiment.log_parameter(name, gin.get_bindings(configurable)[name])

@gin.configurable
def trains(model, data_train, data_test, num_epochs):
    results = []
    for _ in range(num_epochs):
        train(model=model, dataloader=data_train)
    log_gin_parameter(Adam, 'lr')
    log_gin_parameter(RaichuModel, 'H')
    experiment.log_metric('test_loss', evaluate(model, data_train))

def plot_qualities():
    qualities = trains(model=model, data_train=data_train, data_test=data_test)
    train_qualities = list(map(lambda x: x[0], qualities))
    test_qualities = list(map(lambda x: x[1], qualities))
    plt.scatter(list(range(len(train_qualities))), train_qualities, label="train loss")
    plt.scatter(list(range(len(test_qualities))), test_qualities, label="test loss")
    plt.legend()
    plt.show()
    # fig = plt.figure()
    # fig.savefig("lossy2.png")

config_space = []

for lr in np.logspace(-4, 1, num=5):
    for H in [50, 100, 150, 200]:
        config = [
            f"RaichuModel.H = {H}",
            f"Adam.lr = {lr}"
        ] 
        config_space.append(config)

for config in config_space:
    with gin.unlock_config():
        gin.parse_config_files_and_bindings(["config.gin"], bindings=config)
        model=RaichuModel(P=2, K=3)
        trains(model, data_train, data_test)
