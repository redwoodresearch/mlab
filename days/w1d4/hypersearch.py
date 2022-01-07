import gin
from comet_ml import Experiment
import torch
import w1d4_tests
import matplotlib.pyplot as plt
import os

os.system("pip install -r ../../requirements.txt")


class MyModule(torch.nn.Module):
    def __init__(self, P, H, K) -> None:
        super().__init__()
        self.P = P
        self.H = H
        self.K = K

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(P, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, K),
        )

    def forward(self, x):
        return self.layers(x)


def train(model, dataloader, lr, momentum):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = torch.nn.L1Loss()
    for input, target in dataloader:
        optimizer.zero_grad()
        preds = model(input)
        loss = loss_fn(preds, target)
        loss.backward()
        optimizer.step()
    return model


def evaluate(model, dataloader):
    loss_fn = torch.nn.L1Loss()
    loss = 0
    for input, target in dataloader:
        pred = model(input)
        loss += loss_fn(pred, target)
    return loss / len(dataloader)


@gin.configurable
def training_loop(data_train, data_test, lr, momentum, epochs, hidden_size):
    model = MyModule(2, hidden_size, 3)
    train_loss = []
    test_loss = []
    for i in range(epochs):
        train(model, data_train, lr=lr, momentum=momentum)
        train_loss.append(evaluate(model, data_train).item())
        test_loss.append(evaluate(model, data_test).item())
        experiment.log_metric("train_loss", train_loss[i])
        experiment.log_metric("test_loss", test_loss[i])


data_train, data_test = w1d4_tests.load_image("pic.jpg")

config = os.getenv("gin_config")
with gin.unlock_config():
    gin.parse_config(config)
    experiment = Experiment(
        api_key="Ch9OIxODMWCZuK2LSscvbculp",
        project_name="test",
        workspace="pranavgade20",
        log_code=True,
    )

    training_loop(data_train, data_test)

    experiment.end()
