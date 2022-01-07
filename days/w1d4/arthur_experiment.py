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
# else:
    # experiment_params = ben_experiment_params

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

@gin.configurable
def trains(model, data_train, data_test, num_epochs):
    for _ in tqdm(range(num_epochs)):
        train(model=model, dataloader=data_train)
    return evaluate(model, data_train)

config_space = []

for lr in np.logspace(-4, -2, num=3):
    for H in [50, 100, 150, 200]:
        config = [
            f"RaichuModel.H = {H}",
            f"Adam.lr = {lr}"
        ] 
        config_space.append(config)

# for config in config_space:
#     with gin.unlock_config():
#         gin.parse_config_files_and_bindings(["config.gin"], bindings=config)
#         experiment = Experiment(**experiment_params)
#         log_lr = np.log(gin.get_bindings(Adam)['lr'])
#         experiment.log_parameter('log_lr', log_lr)
#         hidden_size = gin.get_bindings(RaichuModel)['H']
#         experiment.log_parameter('hidden_size', hidden_size)
#         model = RaichuModel(P=2, K=3)
#         test_loss = trains(model, data_train, data_test)
#         experiment.log_metric('test_loss', test_loss)
#         experiment.end()

from time import ctime
start_time = ctime()

# for config in config_space:
#     with gin.unlock_config():
#         arthur_experiment_params = {
#             'api_key': "cWQ5thtmlU2pZH62GZVFghchU",
#             'project_name': start_time,
#             'workspace': "arthurconmy",
#         }
#         experiment_params = arthur_experiment_params
#         experiment = Experiment(**experiment_params)

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
    
    # plt.imshow(result)
    # plt.draw()
    # plt.show()
    # print(result.shape)
    # result = result.unsqueeze(-1)
    print(result)
    result = result.unsqueeze(-1)
    plt.imsave("file5.png", result)

tens = t.randn(1080, 1920, 3)
tens = tens.unsqueeze(-1)
print(tens.shape)
plt.imsave("file3.png", tens)

gin.parse_config_files_and_bindings(["arthurconfig.gin"], bindings=config)
model=RaichuModel(P=2, K=3)
trains(model, data_train, data_test)
show_my_image(img, model)