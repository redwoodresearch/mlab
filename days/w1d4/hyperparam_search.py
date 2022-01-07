import os
os.system('pip install -r ../../requirements.txt')

import torch as t
import torch
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
import w1d4_tests as tests
import gin
from typing import Dict, List, Any

@gin.configurable
class MyModel(t.nn.Module):
    def __init__(self, P, H, K):
        super().__init__()
        self.model = t.nn.Sequential(t.nn.Linear(P, H),
                                     t.nn.ReLU(),
                                     t.nn.Linear(H, H),
                                     t.nn.ReLU(),
                                     t.nn.Linear(H, K))
    
    def forward(self, x):
        return self.model(x)

@gin.configurable
def train(model, dataloader, lr, momentum, optimizer=None):
    training_loss = []

    if optimizer is None:
        optimizer = t.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = t.nn.MSELoss()


    for input, target in dataloader:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        training_loss.append(loss)
        loss.backward()
        optimizer.step()
    
    return training_loss

@gin.configurable
def evaluate(model, dataloader, eval_loss=None):
    loss_fn = t.nn.MSELoss()
    total_loss = 0
    examples = 0
    with torch.no_grad():
        for input, target in dataloader:
            output = model(input)
            loss = loss_fn(output, target)
            total_loss += loss
            #experiment.log_metric("eval_loss", loss, step=examples)
            examples += 1

    avg_loss = total_loss/examples

    if eval_loss is not None:
        eval_loss.append(avg_loss)
    
    return avg_loss

@gin.configurable
def train_and_plot(model, train_dataloader, test_dataloader, epochs, plot=True, optimizer=None):
    eval_loss = []
    training_loss = []
    for _ in range(epochs):
        training_loss += train(model, train_dataloader)
        evaluate(model, test_dataloader, eval_loss)
    
    batches_per_epoch = len(training_loss)//len(eval_loss)

    if plot:
        plt.plot(training_loss)
        plt.plot(repeat(t.tensor(eval_loss), 'e -> (e b)', b=batches_per_epoch))
        plt.show()

if __name__ == "__main__":
    print(os.listdir())
    fname = "img.jpg"
    data_train, data_test = tests.load_image(fname)

    with gin.unlock_config():
        gin.parse_config(os.environ.get('gin_config'))
        awesomemodel = MyModel(P=2, K=3)
        train_and_plot(awesomemodel, data_train, data_test, 10, plot=False)