import os
from collections import OrderedDict
import torch
from torch import nn
# import w1d4_tests
import matplotlib.pyplot as plt
import gin

from optims import SGD, RMSProp, Adam


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
    fname = "mona.jpg"
    data_train, data_test =  w1d4_tests.load_image(fname)
    model = Net(2, hidden_size, 3)
    train(model, data_train, data_test)


@gin.configurable
# def train(model, dataloader, opt_str, epochs, learning_rate, loss):
def train(model, data_train, data_test, epochs, lr):
    # try:
    #     opt = globals()[opt_str]()
    # except KeyError:
    #     raise ValueError("This is not a valid optimiser")
    opt = Adam(params=model.parameters(), lr=lr)
    for epoch in range(epochs):
        epoch_loss = train_epoch(model, data_train, opt)
        test_loss = evaluate(model, data_test)
        print(f"{epoch}/{epochs}\t train loss={epoch_loss:.3f}\t test loss={test_loss:.3f}")

if __name__ == "__main__":
    with gin.unlock_config():
        # gin.parse_config_file(config_file="config.gin")
        gin.parse_config(eval(os.environ["PARAMS"])["gin_config"])
        run()