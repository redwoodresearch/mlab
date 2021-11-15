import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import days.bert as bert
import pytest

FLOAT_TOLERANCE = 1e-4


def test_relu():
    input = t.random.uniform(-1, 1, (435, 234))
    my_out = bert.relu(input)
    their_out = F.relu(input)
    absdif = t.mean(t.abs(my_out - their_out))
    print(absdif)
    if absdif > FLOAT_TOLERANCE:
        raise AssertionError({"my": my_out, "their": their_out})


if __name__ == "__main__":
    test_relu()
