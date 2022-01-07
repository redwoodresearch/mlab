# import comet_ml at the top of your file
from comet_ml import Experiment


import gin
import torch
import w1d4_tests
from PIL import Image
from torchvision import transforms
import itertools
import numpy as np
import requests
import os

"""
ThreeLayerNN

Hyperparams: No. of hidden layers, Size of hidden layers, L
"""
@gin.configurable
class Model(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers):
        super().__init__()
        print('Initializing')

        hidden_layers = []
        for i in range(num_hidden_layers):
            hidden_layers.extend([torch.nn.Linear(hidden_size,hidden_size),torch.nn.ReLU()])

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size,hidden_size),
            torch.nn.ReLU(),
            *hidden_layers,
            torch.nn.Linear(hidden_size,output_size),
        )
    
    def forward(self, x):
        return self.layers(x)


@gin.configurable(denylist=["train_dataloader", "test_dataloader"])
def train(model, train_dataloader, test_dataloader, loss, num_epochs, lr, alpha, eps, weight_decay, momentum):
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr, alpha, eps, weight_decay, momentum)
    
    if loss == "l1":
        loss_fn = torch.nn.L1Loss()
    
    for i in range(num_epochs):
        for inp, target in train_dataloader:
            optimizer.zero_grad()	# sets param.grad to None for linked params
                            # (this prevents accumulation of gradients)
            output = model(inp)		  # the model’s predictions
            loss = loss_fn(output, target)	  # measure how bad predictions are
            loss.backward()			  # calculate gradients
            optimizer.step()			  # use gradients to update params
        experiment.log_metric("evaluation loss", evaluate(model, test_dataloader))

    return model


def evaluate(model, dataloader):
    model.eval()
    loss_fn = torch.nn.L1Loss(reduction = 'mean')
    overall_loss = 0
    num_batch = 0
    for inp, target in dataloader:
        output = model(inp)		  # the model’s predictions
        overall_loss += loss_fn(output, target).detach()	  # measure how bad predictions are
        num_batch += 1
    
    return (overall_loss/num_batch).item()

def make_grid(possible_values):
    iters = possible_values.values()
    labels = possible_values.keys()
    result = []
    for combo in itertools.product(*iters):
        combo_dictionary = {}
        for i, label in enumerate(labels):
            combo_dictionary[label] = combo[i]
        result.append(combo_dictionary)
    return result


if __name__ == "__main__":
    fname = "/home/ubuntu/mlab/days/w1d4/hst-bw_orig.jpg"
    height, width = 800, 800
    data_train, data_test =  w1d4_tests.load_image(fname)

    # Extracting things from json_dict to run experiment
    COMET_KEY = os.environ["COMET_KEY"]
    HYPERPARAMS = os.environ["gin_config"]

    hyperparams_list = HYPERPARAMS.split('\n')

    # Create an experiment with your api key
    experiment = Experiment(
        api_key=COMET_KEY,
        project_name="mlab remote",
        workspace="msimontaylor",
    )
    experiment.log_parameters({x.split('=')[0]: x.split('=')[1] for x in hyperparams_list})
    
    with gin.unlock_config():
        gin.parse_config_files_and_bindings(["config.gin"], hyperparams_list)
        model = train(train_dataloader=data_train, test_dataloader=data_test)  
