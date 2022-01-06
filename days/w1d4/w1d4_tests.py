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
import w1d4_sol as sol


def _get_moon_data(unsqueeze_y = False):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=int)
    if unsqueeze_y: # better when the training regimen uses l1 loss, rather than x-ent 
        y = y.unsqueeze(-1)
    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)


def _check_equal(tensor1, tensor2):
    if torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-5):
        print("Congrats! You've passed the test.")
    else:
        print("Your module returns different results from the example solution.")


############################################################################

def test_mlp(MLP):
    x = torch.randn(128, 2)
    torch.manual_seed(534)
    mlp = MLP(2, 32, 2)

    torch.manual_seed(534)
    _mlp = sol._MLP(2, 32, 2)

    _check_equal(mlp(x), _mlp(x))


def test_train(train):
    torch.manual_seed(928)
    lr = 0.1
    momentum = 0.5
    X = torch.rand(512, 2)
    Y = torch.rand(512, 3)
    dl = DataLoader(TensorDataset(X, Y), batch_size=128)

    torch.manual_seed(600)
    model = sol._MLP(2, 32, 3)
    _trained_model = sol._train(model, dl, lr=lr, momentum=momentum)

    torch.manual_seed(600)
    model = sol._MLP(2, 32, 3)
    trained_model = train(model, dl, lr=lr, momentum=momentum)

    x = torch.randn(128, 2)
    _check_equal(trained_model(x), _trained_model(x))

def test_accuracy(accuracy):
    dl = _get_moon_data(unsqueeze_y=True)
    model = sol._MLP(2, 32, 1)
    
    model = sol._train(model, dl, lr=0.1, momentum=0.5)
    _acc = sol._accuracy(model, dl)
    acc = accuracy(model, dl)
    _check_equal(torch.Tensor([_acc]), torch.Tensor([acc]))

def test_evaluate(evaluate):
    torch.manual_seed(928)
    X = torch.rand(512, 2)
    Y = torch.rand(512, 3)
    dl = DataLoader(TensorDataset(X, Y), batch_size=128)

    model = sol._MLP(2, 32, 3)
    model = sol._train(model, dl, lr=0.1, momentum=0.5)
    _loss = sol._evaluate(model, dl)
    loss = evaluate(model, dl)
    _check_equal(torch.Tensor([_loss]), torch.Tensor([loss]))


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
        _w_history = sol._opt_rosenbrock(w, **opt_config)

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
        model = sol._MLP(2, 32, 2)
        opt = torch.optim.SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = sol._MLP(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        _check_equal(w0_correct, w0_submitted)



def test_rmsprop(RMSprop):
    test_cases = [
        dict(lr=0.1, alpha=0.9, eps=0.001, weight_decay=0.0, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.5),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = sol._MLP(2, 32, 2)
        opt = torch.optim.RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = sol._MLP(2, 32, 2)
        opt = RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        _check_equal(w0_correct, w0_submitted)


def test_adam(Adam):
    test_cases = [
        dict(lr=0.1, betas=(0.8, 0.95), eps=0.001, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=0.001, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=0.01, weight_decay=0.08),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = sol._MLP(2, 32, 2)
        opt = torch.optim.Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = sol._MLP(2, 32, 2)
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
