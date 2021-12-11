import torch as t
import days.bert as bert
from test_all import allclose
import torch.nn as nn
import torch.nn.functional as F
import days.modules as reference


def test_layer_norm(fn):
    random_weight = t.empty(9).uniform_(0.8, 1.2)
    random_bias = t.empty(9).uniform_(-0.1, 0.1)
    random_input = t.empty(8, 9)
    their_output = reference.layer_norm(random_input, random_weight, random_bias)
    my_output = fn(random_input, random_weight, random_bias)
    allclose(my_output, their_output, "layer norm")
