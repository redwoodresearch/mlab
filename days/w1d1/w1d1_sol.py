import einops
from einops.einops import rearrange, reduce
import torch
from typing import Optional
from day1_tests import test

############################################################################
# Reference solutions


def ex1():
    return [
        einops.rearrange(torch.arange(3, 9), "(h w) -> h w", h=3, w=2),
        einops.rearrange(torch.arange(1, 7), "(h w) -> h w", h=2, w=3),
        einops.rearrange(torch.arange(1, 7), "a -> 1 a 1"),
    ]

def ex2(temp: torch.Tensor):
    assert len(temp) % 7 == 0, "Input length must be a multiple of 7."

    weekly = einops.rearrange(temp, "(h w) -> h w", w=7)
    weekly_means = einops.reduce(temp, "(h 7) -> h", "mean")
    weekly_zeromean = weekly - weekly_means[:, None]
    weekly_normalised = weekly_zeromean / weekly.std(dim=1, keepdim=True)
    return [weekly_means, weekly_zeromean, weekly_normalised]

def ex3(tensor1: torch.Tensor, tensor2: torch.Tensor):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."
    return (tensor1 * tensor2).sum(dim=-1)


def ex4(H: float, W: float, n: int):
    xaxis = torch.linspace(0, H, n + 1)
    xtile = torch.tile(xaxis, dims=(n + 1, 1))
    yaxis = torch.linspace(0, W, n + 1)[:, None]
    ytile = torch.tile(yaxis, dims=(n + 1,))
    return torch.stack([einops.rearrange(xtile, "h w -> (h w)"), einops.rearrange(ytile, "h w -> (h w)")]).T


def ex5(n: int):
    matrix = (rearrange(torch.arange(n), "i->i 1") == torch.arange(n)).float()
    return matrix

def ex6(n: int, probs: torch.Tensor):
    assert abs(probs.sum() - 1.0) < 0.001
    return (torch.rand((n, 1)) > torch.cumsum(probs, dim=0)).sum(dim=-1)

def ex7(scores: torch.Tensor, y: torch.Tensor):
    #print(f"boutta break, ret is {(scores.argmax(dim=1) == y).to(float).mean()}, tyshapepe {(scores.argmax(dim=1) == y).to(float).mean().shape}")
    return (scores.argmax(dim=1) == y).to(float).mean()

def ex8(scores: torch.Tensor, y: torch.Tensor, k: int):
    return (torch.argsort(scores)[:, -k:] == y[:, None]).any(dim=-1).to(float).mean()


def ex9(prices: torch.Tensor, items: torch.Tensor):
    return torch.gather(prices, 0, items.to(int)).sum()

def ex10(A: torch.Tensor, N: int):
    index = torch.randint(A.shape[-1], (A.shape[0], N))
    return torch.gather(A, 1, index)

def ex11(T: torch.Tensor, K: int, values: Optional[torch.Tensor] = None):
    if values is None:
        values = torch.ones(T.shape[0])
    onehot = torch.zeros(T.shape + (K,))
    return onehot.scatter(-1, T.to(int).unsqueeze(-1), values.unsqueeze(-1))


def relu(tensor: torch.FloatTensor) -> torch.Tensor:
    tensor = tensor.clone()
    tensor[tensor < 0] = 0
    return tensor
ex12 = relu

def dropout(tensor: torch.FloatTensor, drop_fraction: float, is_train: bool):
    if is_train:
        mask = torch.rand_like(tensor) > drop_fraction
        return mask * tensor / (1 - drop_fraction)
    return tensor
ex13 = dropout


def linear(tensor: torch.FloatTensor, weight: torch.FloatTensor, bias: Optional[torch.FloatTensor]):
    x = torch.einsum("...j,kj->...k", tensor, weight)
    if bias is not None:
        x += bias
    return x
ex14 = linear


def layer_norm(x: torch.FloatTensor, reduce_dims, weight: torch.FloatTensor, bias: torch.FloatTensor):
    red_dim_indices = list(range(len(x.shape) - len(reduce_dims), len(x.shape)))
    xmean = x.mean(dim=red_dim_indices, keepdim=True)
    var = ((x - xmean) ** 2).mean(dim=red_dim_indices, keepdim=True)
    xnorm = (x - xmean) / var.sqrt()
    return xnorm * weight + bias
ex15 = layer_norm


def embed(x: torch.LongTensor, embeddings: torch.FloatTensor):
    return embeddings[x]
ex16 = embed

def softmax(tensor: torch.FloatTensor):
    exps = torch.exp(tensor)
    return exps / exps.sum(dim=1, keepdim=True)
ex17 = softmax

def logsoftmax(tensor: torch.FloatTensor):
    C = tensor.max(dim=1, keepdim=True).values
    return tensor - C - (tensor - C).exp().sum(dim=1, keepdim=True).log()
ex18 = logsoftmax


def cross_entropy_loss(logits: torch.FloatTensor, y: torch.LongTensor):
    logprobs = logsoftmax(logits)
    return -torch.gather(logprobs, 1, y[:, None]).mean()
ex19 = cross_entropy_loss

def ex20(x, weight):
    kernel_size = weight.shape[0]
    S = x.shape[0]
    output_length = S - kernel_size + 1
    strided_input = torch.as_strided(x, (output_length, kernel_size), (1, 1), 0)
    added = strided_input * weight
    summed = reduce(added, "s k -> s", "sum")
    return summed

# def ex20_2(v, w):
#     outlen = len(v) - len(w) + 1
#     reps = repeat(t.arange(len(w)), 'w -> h w', h = outlen)
#     reps += t.arange(outlen)[:, None]
#     return einsum("...ij, ...j -> ...i", [v[reps], w])

def ex21(x, weight):
    x = rearrange(x, "b c s -> b s c")
    out_channels, _, kernel_size = weight.shape
    B, S, C = x.shape
    x = x.contiguous()
    strided_input = torch.as_strided(x, (B, S - kernel_size + 1, kernel_size, C, out_channels), (S * C, C, C, 1, 0), 0)
    weight_rearranged = rearrange(weight, "o i k -> k i o")
    added = strided_input * weight_rearranged
    summed = reduce(added, "b s c in_channels out_channels -> b s out_channels", "sum")
    summed = rearrange(summed, "b s c->b c s")
    return summed

# def ex21_2(v, w):
#     batch_size, in_chans, seq_len = v.shape
#     outlen = seq_len - w.shape[-1] + 1
#     u = v.as_strided((batch_size, outlen, in_chans, w.shape[-1]), (in_chans * seq_len, 1, seq_len, 1))
#     return einsum("...tik, oik -> ...ot", [u, w])

def ex22(v, w, padding = 0, stride = 1):
    batch_size, in_chans, seq_len = v.shape
    kernel_size = w.shape[-1]
    outlen = int((seq_len - kernel_size + padding * 2)/stride + 1)
    v = torch.cat([torch.zeros(batch_size, in_chans, padding), v, torch.zeros(batch_size, in_chans, padding)], dim = -1)
    u = v.as_strided((batch_size, outlen, in_chans, kernel_size), (in_chans * seq_len, stride, seq_len, 1))
    return torch.einsum("btik, oik -> bot", [u, w])


for i in range(1, 22):
    f = globals().get(f"ex{i}")
    test(f, i)
