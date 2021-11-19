import math

import torch as t
import numpy as np
from torch.nn import Module, Parameter

from utils import tpeek


def softmax(tensor: t.Tensor, dim: int = 0, eps=1e-9):
    exps = np.e ** tensor
    exp_sums = exps.sum(dim=dim, keepdim=True)
    result = exps / (exp_sums)
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


class ReLU(Module):
    def forward(self, x):
        return relu(x)


class LayerNorm(Module):
    def __init__(self, shape, eps=1e-6):
        if isinstance(shape, int):
            shape = (shape,)

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


def cross_entropy(input, target, ignore_index=None, max=1e12):
    exps = np.e ** input
    exps[exps > max] = max
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


def _ntuple(n):
    import collections
    from itertools import repeat

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        super().__init__()
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        weight_size = (out_channels, in_channels // groups, *kernel_size)
        bound = 1 / math.sqrt(weight_size[1] * math.prod(kernel_size))
        self.weight = Parameter(t.FloatTensor(*weight_size).uniform_(-bound, bound))

        if bias:
            fan_in = math.prod(self.weight.shape[1:])
            bound = 1 / math.sqrt(fan_in)
            self.bias = Parameter(t.FloatTensor(out_channels).uniform_(-bound, bound))
        else:
            self.bias = t.zeros(out_channels)

    def forward(self, x):
        sH, sW = self.stride
        pH, pW = self.padding
        B, iC, iH, iW = x.shape
        oC, _, kH, kW = self.weight.shape
        oH = (iH + 2 * pH - kH) // sH + 1
        oW = (iW + 2 * pW - kW) // sW + 1

        from torch.nn.functional import pad

        padded_x = pad(x, [pH, pH, pW, pW])

        sx_stride = t.tensor(padded_x.stride())
        sx_stride[-2] *= sH
        sx_stride[-1] *= sW
        sx_stride = tuple(sx_stride)

        strided_x = t.as_strided(padded_x, size=(B, iC, oH, oW, kH, kW), stride=sx_stride + (iW + 2 * pW, 1))
        return t.einsum("bcxyij,ocij->boxy", strided_x, self.weight) + self.bias.reshape(1, -1, 1, 1)


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=1,
        dilation=1,
    ):
        super().__init__()
        if stride is None:
            stride = kernel_size

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

    def forward(self, x):
        sH, sW = self.stride
        pH, pW = self.padding
        B, iC, iH, iW = x.shape
        kH, kW = self.kernel_size
        oH = (iH + 2 * pH - kH) // sH + 1
        oW = (iW + 2 * pW - kW) // sW + 1

        from torch.nn.functional import pad

        padded_x = pad(x, [pH, pH, pW, pW])

        sx_stride = t.tensor(padded_x.stride())
        sx_stride[-2] *= sH
        sx_stride[-1] *= sW
        sx_stride = tuple(sx_stride)

        strided_x = t.as_strided(padded_x, size=(B, iC, oH, oW, kH, kW), stride=sx_stride + (iW + 2 * pW, 1))
        return strided_x.amax((-2, -1))


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        return x.flatten(self.start_dim, self.end_dim)


class BatchNorm2d(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(t.ones(num_features))
        self.bias = Parameter(t.zeros(num_features))
        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x):
        ids = (0, 2, 3)
        if self.training:
            mean = x.mean(ids)
            var = x.var(ids, unbiased=False)
            a = self.momentum
            self.running_mean.data = (1 - a) * self.running_mean.data + a * mean
            self.running_var.data = (1 - a) * self.running_var.data + a * var
            self.num_batches_tracked.data += 1
        else:
            mean = self.running_mean
            var = self.running_var

        rs = lambda u: u.reshape(1, -1, 1, 1)
        return rs(self.weight) * (x - rs(mean)) / t.sqrt(rs(var) + self.eps) + rs(self.bias)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = _pair(output_size)

    def forward(self, x):
        def kernels(in_dim, out_dim):
            return [slice(math.floor((i * in_dim) / out_dim),
                          math.ceil(((i + 1) * in_dim) / out_dim))
                    for i in range(out_dim)]

        B, C, iH, iW = x.shape
        oH, oW = self.output_size

        kHs = kernels(iH, oH)
        kWs = kernels(iW, oW)

        out = t.empty((B, C, oH, oW))
        for i, ker_H in enumerate(kHs):
            for j, ker_W in enumerate(kWs):
                out[:, :, i, j] = t.mean(x[:, :, ker_H, ker_W], (-2, -1))
        return out
