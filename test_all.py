import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import activation
from transformers.utils.dummy_sentencepiece_objects import PegasusTokenizer
import days.bert as bert
import pytest

MAX_TOLERANCE = 5e-3
AVG_TOLERANCE = 1e-4


def tpeek(name, tensor):
    print(f"{name} {t.mean(tensor).item()} {t.var(tensor).item()} {t.flatten(tensor)[:10].cpu().tolist()}")


def tstat(name, tensor):
    print(name, "mean", "{0:.4g}".format(t.mean(tensor).item()), "var", "{0:.4g}".format(t.var(tensor).item()))


def init_both(my_class, their_class, *args, **kwargs):
    t.random.manual_seed(0)
    my_class = my_class(*args, **kwargs)
    t.random.manual_seed(0)
    their_class = their_class(*args, **kwargs)
    return my_class, their_class


def testy(my_out, their_out, name):
    maxdif = t.max(t.abs(my_out - their_out))
    avgdif = t.mean(my_out - their_out)
    if maxdif > MAX_TOLERANCE or avgdif > AVG_TOLERANCE:
        tpeek("mine", my_out)
        tpeek("theirs", their_out)
        raise AssertionError(f"{name} avgdif {float(avgdif)} maxdif {float(maxdif)}")
    print(f"{name} avgdif {float(avgdif)} maxdif {float(maxdif)}")


def test_relu():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = bert.relu(input)
    their_out = F.relu(input)
    testy(my_out, their_out, "relu")


def test_gelu():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = bert.gelu(input)
    their_out = F.gelu(input)
    testy(my_out, their_out, "gelu")


def test_softmax():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = bert.softmax(input, dim=1)
    their_out = F.softmax(input, dim=1)
    testy(my_out, their_out, "softmax")


def test_normalize():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    my_out = bert.normalize(input, dim=1)
    their_out = F.normalize(input, dim=1)
    testy(my_out, their_out, "normalize")


def test_layer_norm():
    input = t.FloatTensor(435, 234).uniform_(-10, 10)
    t.random.manual_seed(0)
    my_layer_norm = bert.LayerNorm((234,))
    t.random.manual_seed(0)
    their_layer_norm = nn.LayerNorm(234)

    my_out = my_layer_norm(input)
    their_out = their_layer_norm(input)
    testy(my_out, their_out, "layer_norm")


def test_linear():
    input = t.FloatTensor(435, 435, 234).uniform_(-10, 10)
    my_linear, their_linear = init_both(bert.Linear, nn.Linear, 234, 111)
    tstat("my weight", my_linear.weight)
    tstat("their weight", their_linear.weight)

    my_out = my_linear(input)
    their_out = their_linear(input)
    testy(my_out, their_out, "linear")


def test_embedding():
    unembed_input = t.FloatTensor(643, 23, 111).uniform_(-1, 1)
    embed_input = t.LongTensor([1, 2, 3])
    my_embedding, their_embedding = init_both(bert.Embedding, nn.Embedding, 234, 111)
    tstat("my embedding", my_embedding.embedding)
    tstat("their embedding", their_embedding.weight)

    my_out = my_embedding.embed(embed_input)
    their_out = their_embedding(embed_input)
    testy(my_out, their_out, "embedding")

    my_out = my_embedding.unembed(unembed_input)


def test_self_attention_fundamentals():
    width = 16
    activations = t.FloatTensor(10, 10, width).normal_(0, 1)
    q = t.eye(width)
    k = t.eye(width)
    v = t.eye(width)
    tpeek("activations", activations)
    output = bert.multi_head_self_attention(
        activations, attention_masks=None, num_heads=4, project_query=q, project_key=k, project_value=v
    )
    print(output)


if __name__ == "__main__":
    test_relu()
    test_gelu()
    test_softmax()
    test_normalize()
    test_layer_norm()
    test_embedding()
    test_self_attention_fundamentals()
    # test_linear()  # idk why this isn't producing same result
