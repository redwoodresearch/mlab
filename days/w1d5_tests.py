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


def _get_moon_data():
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=int)
    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)


def _check_equal(tensor1, tensor2):
    if torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-5):
        print("Congrats! You've passed the test.")
    else:
        print("Your module returns different results from the example solution.")


############################################################################


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


def test_mlp(MLP):
    x = torch.randn(128, 2)
    torch.manual_seed(534)
    mlp = MLP(2, 32, 2)

    torch.manual_seed(534)
    _mlp = _MLP(2, 32, 2)

    _check_equal(mlp(x), _mlp(x))


def _train(model: nn.Module, dataloader: DataLoader, lr, momentum):
    opt = torch.optim.SGD(model.parameters(), lr, momentum)
    for X, y in dataloader:
        opt.zero_grad()
        pred = model(X)
        loss = F.l1_loss(pred, y)
        loss.backward()
        opt.step()
    return model


def test_train(train):
    torch.manual_seed(928)
    lr = 0.1
    momentum = 0.5
    X = torch.rand(512, 2)
    Y = torch.rand(512, 3)
    dl = DataLoader(TensorDataset(X, Y), batch_size=128)

    torch.manual_seed(600)
    model = _MLP(2, 32, 3)
    _trained_model = _train(model, dl, lr=lr, momentum=momentum)

    torch.manual_seed(600)
    model = _MLP(2, 32, 3)
    trained_model = train(model, dl, lr=lr, momentum=momentum)

    x = torch.randn(128, 2)
    _check_equal(trained_model(x), _trained_model(x))


def _accuracy(model: nn.Module, dataloader: DataLoader):
    n_correct = 0
    n_total = 0
    for X, y in dataloader:
        n_correct += (model(X).argmax(1) == y).sum().item()
        n_total += len(y)
    return n_correct / n_total


def test_accuracy(accuracy):
    dl = _get_moon_data()
    model = _MLP(2, 32, 2)
    model = _train(model, dl, lr=0.1)
    _acc = _accuracy(model, dl)
    acc = accuracy(model, dl)
    _check_equal(torch.Tensor([_acc]), torch.Tensor([acc]))


def _evaluate(model: nn.Module, dataloader: DataLoader):
    sum_abs = 0.0
    n_elems = 0
    for X, y in dataloader:
        sum_abs += (model(X) - y).abs().sum()
        n_elems += y.shape[0] * y.shape[1]
    return sum_abs / n_elems


def test_evaluate(evaluate):
    torch.manual_seed(928)
    X = torch.rand(512, 2)
    Y = torch.rand(512, 3)
    dl = DataLoader(TensorDataset(X, Y), batch_size=128)

    model = _MLP(2, 32, 3)
    model = _train(model, dl, lr=0.1, momentum=0.5)
    _loss = _evaluate(model, dl)
    loss = evaluate(model, dl)
    _check_equal(torch.Tensor([_loss]), torch.Tensor([loss]))


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


def test_rosenbrock(opt_rosenbrock):
    test_cases = [
        dict(lr=0.001, momentum=0.0, n_iter=10),
        dict(lr=0.001, momentum=0.8, n_iter=20),
    ]
    for opt_config in test_cases:
        w = torch.Tensor([-1.5, 2.5])
        w.requires_grad = True
        w_history = opt_rosenbrock(w, **opt_config)

        w = torch.Tensor([-1.5, 2.5])
        w.requires_grad = True
        _w_history = _opt_rosenbrock(w, **opt_config)

        print("\nTesting configuration: ", opt_config)
        _check_equal(w_history, _w_history)

    
############################################################################


def _train_with_opt(model, opt):
    dl = _get_moon_data()
    for i, (X, y) in enumerate(dl):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()


class _SGD:
    def __init__(self, params, lr: float, momentum: float, dampening: float, weight_decay: float):
        self.params = list(params)
        self.lr = lr
        self.wd = weight_decay
        self.mu = momentum
        self.tau = dampening
        self.b = [torch.zeros_like(p) for p in self.params]

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


def test_sgd(SGD):
    test_cases = [
        dict(lr=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0),
        dict(lr=0.1, momentum=0.7, dampening=0.0, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, dampening=0.5, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, dampening=0.5, weight_decay=0.05),
        dict(lr=0.2, momentum=0.8, dampening=0.0, weight_decay=0.05),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = torch.optim.SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        _check_equal(w0_correct, w0_submitted)


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


def test_rmsprop(RMSprop):
    test_cases = [
        dict(lr=0.1, alpha=0.9, eps=0.001, weight_decay=0.0, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.5),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = torch.optim.RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        _check_equal(w0_correct, w0_submitted)


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


def test_adam(Adam):
    test_cases = [
        dict(lr=0.1, betas=(0.8, 0.95), eps=0.001, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=0.001, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=0.01, weight_decay=0.08),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = torch.optim.Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        _check_equal(w0_correct, w0_submitted)


##################################################################################


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


def plot_image(fname):
    img = Image.open(fname)
    fig = plt.imshow(img)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
