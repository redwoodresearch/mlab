import torch as t
import numpy as np
from torch.nn import Module, Parameter

from utils import tpeek


def softmax(tensor: t.Tensor, dim: int = 0):
    exps = np.e ** tensor
    exp_sums = exps.sum(dim=dim, keepdim=True)
    result = exps / exp_sums
    return result


def relu(tensor: t.Tensor) -> t.Tensor:
    tensor[tensor < 0] = 0
    return tensor


# gelu from openai github, not the same as torch's
def gelu(x):
    return 0.5 * x * (1 + t.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * t.pow(x, 3))))


# TODO: figure out what this should actually be
def normalize(tensor: t.Tensor, dim: int = -1, eps=1e-12):
    norm = t.norm(tensor, dim=dim, keepdim=True)
    norm[norm < eps] = eps
    tensor = tensor / norm
    return tensor


class LayerNorm(Module):
    def __init__(self, shape, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.bias = Parameter(t.zeros(shape))
        self.weight = Parameter(t.ones(shape))
        self.eps = eps
        # indexes to normalize over
        self.idx_list = [-i - 1 for i, _ in enumerate(shape)]

    def forward(self, tensor):
        tensor = (tensor - tensor.mean(*self.idx_list, keepdim=True)) / t.sqrt(
            tensor.var(*self.idx_list, keepdim=True) + self.eps
        )
        tensor = tensor * self.weight + self.bias
        return tensor


class Dropout(Module):
    def __init__(self, fraction=0.1):
        super(Dropout, self).__init__()
        self.fraction = fraction

    def forward(self, input):
        if self.training:
            mask = t.empty_like(input).uniform_(0, 1) > self.fraction
            return mask * input
        return input


class Linear(Module):
    def __init__(self, x, y, bias=True):
        super(Linear, self).__init__()
        weight_bound = 1 / np.sqrt(x)
        self.weight = Parameter(t.FloatTensor(y, x).uniform_(-weight_bound, weight_bound))
        if bias:
            bias_bound = 1 / np.sqrt(y)
            self.bias = Parameter(t.FloatTensor(y).uniform_(-bias_bound, bias_bound))
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = t.einsum("...j,kj->...k", x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x


class Embedding(Module):
    def __init__(self, vocab_size: int, embedding_size: int):
        super(Embedding, self).__init__()
        self.weight = Parameter(t.FloatTensor(vocab_size, embedding_size).normal_(0, 1))
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

    def forward(self, ids):
        return self.weight[ids]

    def unembed(self, embeddings):
        # because the weight initialization is meant for embedding, we need to scale it when we matmul
        return t.einsum("...j,kj->...k", embeddings, self.weight) / np.sqrt(self.embedding_size)


def cross_entropy(input, target, ignore_index=None):
    exps = np.e ** input
    exp_sums = exps.sum(dim=-1)
    exp_sum_logs = t.log(exp_sums)
    gathered = t.gather(input, -1, target.unsqueeze(-1)).squeeze(-1)
    token_losses = exp_sum_logs - gathered
    if ignore_index is not None:
        live_mask = target != ignore_index
        token_losses *= live_mask
        live_fraction = t.sum(live_mask) / live_mask.nelement()
        if live_fraction == 0:
            return t.FloatTensor(0)
        token_losses /= live_fraction
    return t.mean(token_losses)
