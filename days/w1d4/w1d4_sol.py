import einops
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import make_moons
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from torchvision import transforms
from typing import Tuple
import w1d4_tests as tests


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


def _train(model: nn.Module, dataloader: DataLoader, lr, momentum):
    opt = torch.optim.SGD(model.parameters(), lr, momentum)
    for X, y in dataloader:
        opt.zero_grad()
        pred = model(X)
        # print(pred.shape, y.shape)
        #assert False
        loss = F.l1_loss(pred, y)
        loss.backward()
        opt.step()
    return model


def _accuracy(model: nn.Module, dataloader: DataLoader):
    n_correct = 0
    n_total = 0
    for X, y in dataloader:
        n_correct += (model(X).argmax(1) == y).sum().item()
        n_total += len(y)
    return n_correct / n_total


def _evaluate(model: nn.Module, dataloader: DataLoader):
    sum_abs = 0.0
    n_elems = 0
    for X, y in dataloader:
        sum_abs += (model(X) - y).abs().sum()
        n_elems += y.shape[0] * y.shape[1]
    return sum_abs / n_elems

def _rosenbrock(x, y, a=1, b=100):
    return (a-x)**2 + b*(y-x**2)**2 + 1

def _opt_rosenbrock(xy, lr, momentum, n_iter):
    w_history = torch.zeros([n_iter+1, 2])
    w_history[0] = xy.detach()
    opt = torch.optim.SGD([xy], lr=lr, momentum=momentum)
    
    for i in range(n_iter):
        opt.zero_grad()
        _rosenbrock(xy[0], xy[1]).backward()
        
        opt.step()
        w_history[i+1] = xy.detach()
    return w_history



class _SGD:
    def __init__(self, params, lr: float, momentum: float, dampening: float, weight_decay: float):
        self.params = list(params)
        self.lr = lr
        self.wd = weight_decay
        self.mu = momentum
        self.tau = dampening
        self.b = [None for _ in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                if self.mu:
                    if self.b[i] is not None:
                        self.b[i] = self.mu * self.b[i] + (1.0 - self.tau) * g
                    else:
                        self.b[i] = g
                    g = self.b[i]
                p -= self.lr * g


class _RMSprop:
    def __init__(
        self, params, lr: float, alpha: float, eps: float, weight_decay: float,
            momentum: float):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.wd = weight_decay
        self.mu = momentum

        self.v = [torch.zeros_like(p) for p in self.params]
        self.b = [torch.zeros_like(p) for p in self.params]
        self.g_ave = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * g ** 2
                if self.mu:
                    self.b[i] = self.mu * self.b[i] + g / (self.v[i].sqrt() + self.eps)
                    p -= self.lr * self.b[i]
                else:
                    p -= self.lr * g / (self.v[i].sqrt() + self.eps)


class _Adam:
    def __init__(self, params, lr: float, betas: Tuple[float, float], eps: float, weight_decay: float):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g ** 2
                mhat = self.m[i] / (1.0 - self.beta1 ** self.t)
                vhat = self.v[i] / (1.0 - self.beta2 ** self.t)
                p -= self.lr * mhat / (vhat.sqrt() + self.eps)

if __name__ == "__main__":
    tests.test_mlp(_MLP)
    tests.test_train(_train)
    tests.test_accuracy(_accuracy)
    tests.test_evaluate(_evaluate)
    tests.test_rosenbrock(_opt_rosenbrock)
    tests.test_sgd(_SGD)
    tests.test_rmsprop(_RMSprop)
    tests.test_adam(_Adam)

