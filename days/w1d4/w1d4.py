# import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="72XQSdnwnBcob4Q8NpbJHewll",
    project_name="w1d4 gridsearch",
    workspace="danielb",
)

import os
from collections import OrderedDict
import torch
from torch import nn
# import w1d4_tests
import matplotlib.pyplot as plt
import gin
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import einops
from optims import SGD, RMSProp, Adam


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


class Net(nn.Module):
    def __init__(self, P, H, K):
        super(Net, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('lin1', nn.Linear(P, H)),
            ('relu1', nn.ReLU()),
            ('lin2', nn.Linear(H, H)),
            ('relu2', nn.ReLU()),
            ('lin3', nn.Linear(H, K)),
        ]))
    
    def forward(self, x):
        return self.net(x)

def train_epoch(model, dataloader, opt):
    model.train()
    acc_loss = 0
    for (x, y) in dataloader:
        model.zero_grad()
        pred = model(x)
        loss = torch.abs(y - pred).mean()
        loss.backward()
        acc_loss += loss
        opt.step()
    return (acc_loss/len(dataloader)).item()

def evaluate(model, dataloader):
    model.eval()
    tot_loss = 0
    for x, y in dataloader:
        pred = model(x)
        tot_loss += (pred-y).abs().mean()
    return (tot_loss / len(dataloader)).item()

# epochs = 50
# model = Net(2, 400, 3)
# opt = Adam(model.parameters(), lr=0.01)

# training_loss = []
# eval_loss = []
# for epoch in range(epochs):
#     training_loss.append(train(model, data_train, opt=opt))
#     eval_loss.append(evaluate(model, data_test))

# plt.plot(training_loss, label="train")
# plt.plot(eval_loss, label="eval")
# plt.legend()
# def setup_optim(opt_str, params, lr, )

@gin.configurable
def run(hidden_size):
    experiment.log_parameter("hidden size", hidden_size)
    fname = "days/w1d4/mona.jpg"
    data_train, data_test = load_image(fname)
    model_file = "days/w1d4/model.pt"
    model = Net(2, hidden_size, 3)
    train(model, data_train, data_test)
    torch.save(model.state_dict(), model_file)
    experiment.log_model("OurNet", model_file)


@gin.configurable
# def train(model, dataloader, opt_str, epochs, learning_rate, loss):
def train(model, data_train, data_test, epochs, lr):
    # try:
    #     opt = globals()[opt_str]()
    # except KeyError:
    #     raise ValueError("This is not a valid optimiser")
    opt = Adam(params=model.parameters(), lr=lr)
    experiment.log_parameter("lr", lr)
    for epoch in range(epochs):
        epoch_loss = train_epoch(model, data_train, opt)
        test_loss = evaluate(model, data_test)
        experiment.log_metric("train epoch loss", epoch_loss, epoch=epoch)
        experiment.log_metric("test epoch loss", test_loss, epoch=epoch)
        print(f"{epoch}/{epochs}\t train loss={epoch_loss:.3f}\t test loss={test_loss:.3f}")

if __name__ == "__main__":
    with gin.unlock_config():
        # gin.parse_config_file(config_file="config.gin")
        # print("in w1d4", os.environ)
        gin.parse_config(eval(os.environ["PARAMS"])["gin_config"])
        run()