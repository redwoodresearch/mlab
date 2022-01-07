from comet_ml import Experiment
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import gin
import time
import numpy as np
import os
import json

import einops
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

def load_image(fname, n_train=8192, batch_size=128):
    img = Image.open(fname)
    tensorize = transforms.ToTensor()
    img = tensorize(img)
    img = einops.rearrange(img, "c h w -> h w c")
    height, width = img.shape[:2]

    n_trn = n_train
    n_tst = 1024
    X1 = torch.randint(0, height, (n_trn + n_tst,))
    X2 = torch.randint(0, width, (n_trn + n_tst,))
    X = torch.stack([X1.float() / height - 0.5, X2.float() / width - 0.5]).T
    Y = img[X1, X2] - 0.5

    Xtrn, Xtst = X[:n_trn], X[n_trn:]
    Ytrn, Ytst = Y[:n_trn], Y[n_trn:]

    dl_trn = DataLoader(TensorDataset(Xtrn, Ytrn), batch_size=batch_size, shuffle=True)
    dl_tst = DataLoader(TensorDataset(Xtst, Ytst), batch_size=batch_size)
    return dl_trn, dl_tst

class MLP(nn.Module):
    def __init__(self, P, H, K):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(P, H)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(H, H)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(H, K)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x
    
def evaluate(model, dataloader):
    model.eval()
    losses = []
    for input, target in dataloader:
        output = model(input)
        with torch.no_grad():
            loss = F.l1_loss(output, target)
            losses.append(loss.item())
    return sum(losses)/len(losses)

class SGD():
    def __init__(self, params, lr=0.0001, momentum=0.9, dampening=0, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.t = 0
        self.prev_b = [None for _ in range(len(self.params))] 
   
    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        b = None
        for i, p_i in enumerate(self.params):
            with torch.no_grad():          
                g = p_i.grad
                if self.weight_decay != 0:
                    g += self.weight_decay* p_i
                if self.momentum != 0:
                    if self.t == 0:
                        b = g
                    else:
                        b = self.momentum * self.prev_b[i] + (1 - self.dampening) * g
                    g = b
                self.prev_b[i] = b
                p_i -= self.lr * g
        self.t += 1
        
class RMSProp():
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.prev_b = [0 for _ in range(len(self.params))] 
        self.prev_v = [0 for _ in range(len(self.params))] 
        
    def zero_grad(self):
        for param in self.params:
            param.grad = None
        
    def step(self):
        with torch.no_grad():
            for i, p_i in enumerate(self.params):
                g = p_i.grad
                if self.weight_decay != 0:
                    g += self.weight_decay * p_i
                v = self.alpha * self.prev_v[i] + (1 - self.alpha) * g**2
                self.prev_v[i] = v
                if self.momentum > 0:
                    b = self.momentum * self.prev_b[i] + g / (torch.sqrt(v) + self.eps)
                    self.prev_b[i] = b
                    p_i -= self.lr * b
                else:
                    p_i -= (self.lr * g)/(torch.sqrt(v) + self.eps)
                    
                    
class Adam():
    def __init__(self, params, lr=0.01, betas=[0.9,0.999], eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.prev_m = [0 for _ in range(len(self.params))] 
        self.prev_v = [0 for _ in range(len(self.params))] 
        self.t = 1

    def zero_grad(self):
        for param in self.params:
            param.grad = None
        
    def step(self):
        with torch.no_grad():
            for i, p_i in enumerate(self.params):
                g = p_i.grad
                if self.weight_decay != 0:
                    g += self.weight_decay * p_i
                m = self.betas[0] * self.prev_m[i] + (1 - self.betas[0]) * g
                v = self.betas[1] * self.prev_v[i] + (1 - self.betas[1]) * g**2
                self.prev_m[i] = m
                self.prev_v[i] = v
                m_hat = m / (1 - self.betas[0] ** self.t)
                v_hat = v / (1 - self.betas[1] ** self.t)          
                p_i -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        self.t += 1
        
    
@gin.configurable
def newtrain(learning_rate, momentum, epochs, optimizer, loss, hidden_size, weight_decay):
    IMAGE_NAME = "days/w1d4/cat.jpg"
    model = MLP(2, hidden_size, 3)
    data_train, data_test =  load_image(IMAGE_NAME)
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "rmsprop":
        optimizer = RMSProp(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    if loss == "mae":
        loss_f = F.l1_loss
    else:
        loss_f = nn.MSELoss()
    training_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        _training_losses = []
        start_time = time.time()
        for input, target in data_train:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_f(output, target)
            _training_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        end_time = time.time()
        experiment.log_metric("time", end_time - start_time, epoch=epoch)
        test_loss = evaluate(model, data_test)
        experiment.log_metric("test loss", test_loss, epoch=epoch)
        test_losses.append(test_loss)  
        training_losses.append(sum(_training_losses) / len(_training_losses))
    return model, training_losses, test_losses


if __name__ == "__main__":
    with gin.unlock_config():
        experiment = Experiment(
            api_key="vaznwXsdK5Z3Hug3FKZCl9lGN",
            project_name="mlab-w1d4",
            workspace="tomtseng",
        )
        print("Running with the following gin bindings")
        print(os.environ["PARAMS"]) 
        env_params = json.loads(os.environ["PARAMS"])
        config = env_params["gin_config"]
        gin.parse_config_files_and_bindings([], bindings=config)
        # Log the gin bindings of newtrain
        for name, val in gin.get_bindings(newtrain).items():
            if type(val) in [str, int, float]:
                experiment.log_parameter(name, val)
        _, training_losses, test_losses = newtrain()
        print(training_losses, test_losses)
