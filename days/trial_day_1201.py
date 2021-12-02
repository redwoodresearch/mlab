import einops
import math
from einops.einops import rearrange, reduce
import numpy as np
import torch
import typing


def _allclose_tensorlists(tensorlist1, tensorlist2):
    for t1, t2 in zip(tensorlist1, tensorlist2):
        if t1.shape != t2.shape:
            return False
        if not torch.allclose(t1, t2):
            return False
    return True


def test_function(f, ex_num):
    g = globals()
    ex = g.get(f"ex{ex_num}")
    assert ex, f"There's no solution for exercise {ex_num}"
    test_cases = g.get(f"test_cases{ex_num}")

    deterministic_exercises = {2, 3, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20}
    
    if ex_num == 1:
        return _allclose_tensorlists(f(), ex1())
    elif ex_num in deterministic exercises:
        for tc in test_cases():
            if not _allclose_tensorlists(f(*tc), ex(*tc)):
                return False
        return True

    print("Sorry, can't test probabilistic outputs.")    
    

def ex1():
    return [
        einops.rearrange(torch.arange(3, 9), "(h w) -> h w", h=3, w=2),
        einops.rearrange(torch.arange(1, 7), "(h w) -> h w", h=2, w=3),
        einops.rearrange(torch.arange(1, 7), "a -> 1 a 1"),
    ]


def test_cases1():
    return []


def ex2(temp: torch.Tensor):
    assert len(temp) % 7 == 0, "Input length must be a multiple of 7."

    weekly = einops.rearrange(temp, "(h w) -> h w", w=7)
    weekly_means = einops.reduce(temp, "(h 7) -> h", "mean")
    weekly_zeromean = weekly - weekly_means[:, None]
    weekly_normalised = weekly_zeromean / weekly.std(dim=1, keepdim=True)
    return [weekly_means, weekly_zeromean, weekly_normalised]


def test_cases2():
    return [
        torch.Tensor([71, 72, 70, 75, 71, 72, 70, 68, 65, 60, 68, 60, 55, 59, 75, 80, 85, 80, 78, 72, 83]),
        torch.Tensor(
            [
                57.1928,
                63.4550,
                67.5766,
                66.8976,
                51.4928,
                60.0626,
                69.2239,
                73.3777,
                58.3225,
                53.6842,
                72.8498,
                56.8322,
                79.1494,
                72.5929,
                79.7508,
                66.5618,
                54.4778,
                75.0086,
                79.7727,
                72.2138,
                77.9900,
            ]
        ),
        torch.rand((70,)) * 70 + 30,
    ]


def ex3(tensor1: torch.Tensor, tensor2: torch.Tensor):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."
    return (tensor1 * tensor2).sum(dim=-1)


def test_cases3():
    out = []
    for i in range(10):
        n_dims = np.random.randint(2, 5)
        shape = list(np.random.randint(10, 50, n_dims))
        out += [[torch.randn(shape), torch.randn(shape)]]
    return out


def ex4(H: float, W: float, n: int):
    xaxis = torch.linspace(0, H, n + 1)
    xtile = torch.tile(xaxis, dims=(n + 1, 1))
    yaxis = torch.linspace(0, W, n + 1)[:, None]
    ytile = torch.tile(yaxis, dims=(n + 1,))
    return torch.stack([einops.rearrange(xtile, "h w -> (h w)"), einops.rearrange(ytile, "h w -> (h w)")]).T


def test_cases4():
    out = []
    for i in range(10):
        n = np.random.randint(2, 10)
        w = np.random.randint(5, 20)
        h = np.random.randint(5, 20)
        out += [(h * n, w * n, n)]
    return out


def ex5(n: int):
    matrix = (rearrange(torch.arange(n), "i->i 1") == torch.arange(n)).float()

    return matrix


def test_cases5():
    return list(range(2, 11))


def ex6(n: int, probs: torch.Tensor):
    assert abs(probs.sum() - 1.0) < 0.001
    return (torch.rand((n, 1)) > torch.cumsum(probs, dim=0)).sum(dim=-1)


def test_fn_6(fn):
    probs = t.nn.functional.softmax(t.rand(10), dim=-1)
    t = t.stack([fn(probs) for _ in range(1000)], dim=0).sum(0)
    assert t.allclose(probs, t, atol=0.05, rtol=0.1)


def test_cases6():
    out = []
    for i in range(10):
        n = np.random.randint(10, 100)
        k = np.random.randint(2, 10)
        probs = np.random.rand(k)
        out += [(n, probs / probs.sum())]
    return out


def ex7(scores: torch.Tensor, y: torch.Tensor):
    return (scores.argmax(dim=1) == y).to(float).mean()


def test_cases7():
    out = []
    for i in range(10):
        n_inputs = np.random.randint(50, 100)
        n_classes = np.random.randint(5, 20)
        scores = torch.randn((n_inputs, n_classes))
        y = torch.randint(n_classes, (n_inputs,))
        out += [[scores, y]]
    return out


def ex8(scores: torch.Tensor, y: torch.Tensor, k: int):
    return (torch.argsort(scores)[:, -k:] == y[:, None]).any(dim=-1).to(float).mean()


def test_cases8():
    out = []
    for i in range(10):
        n_inputs = np.random.randint(50, 100)
        n_classes = np.random.randint(10, 20)
        scores = torch.randn((n_inputs, n_classes))
        y = torch.randint(n_classes, (n_inputs,))
        k = np.random.randint(2, n_classes // 2)
        out += [[scores, y, k]]
    return out


def ex9(prices: torch.Tensor, items: torch.Tensor):
    return torch.gather(prices, 0, items.to(int)).sum()


def test_cases9():
    out = []
    for i in range(10):
        n_items = np.random.randint(5, 20)
        prices = torch.rand(n_items) * 100
        n_buys = np.random.randint(40, 200)
        items = torch.randint(n_items, (n_buys,))
        out += [[prices, items]]
    return out


def ex10(A: torch.Tensor, N: int):
    index = torch.randint(A.shape[-1], (A.shape[0], N))
    return torch.gather(A, 1, index)


def test_cases10():
    out = []
    for i in range(10):
        m, k, n = np.random.randint(20, 100, (3,))
        A = torch.randn((m, k))
        out += [[A, n]]
    return out


def ex11(T: torch.Tensor, K: int, values: typing.Optional[torch.Tensor] = None):
    if values is None:
        values = torch.ones(T.shape[0])
    onehot = torch.zeros(T.shape + (K,))
    return onehot.scatter(1, T.to(int).unsqueeze(-1), values.unsqueeze(-1))


def test_cases11():
    out = []
    for i in range(10):
        n_dim = np.random.randint(1, 3)
        K = np.random.randint(5, 10)
        shape = list(np.random.randint(2, 10, (n_dim,)))
        values = torch.rand(shape) * 100
        out += [[torch.randint(K, shape), K, values]]
    return out


def relu(tensor: torch.FloatTensor) -> torch.Tensor:
    tensor = tensor.clone()
    tensor[tensor < 0] = 0
    return tensor


def test_cases12():
    out = []
    for i in range(10):
        n_dim = np.random.randint(1, 4)
        shape = list(np.random.randint(5, 10, (n_dim,)))
        out += [torch.randn(shape)]
    return out


def dropout(tensor: torch.FloatTensor, drop_fraction: float, is_train: bool):
    if is_train:
        mask = torch.rand_like(tensor) > drop_fraction
        return mask * tensor / (1 - drop_fraction)
    return tensor


def test_cases13():
    tensors = test_cases12()
    drop_fractions = np.random.rand(10) * 0.6 + 0.2
    is_train = np.random.randint(0, 2, (10,)).astype(bool)
    return list(zip(tensors, drop_fractions, is_train))


def linear(tensor: torch.FloatTensor, weight: torch.FloatTensor, bias: typing.Optional[torch.FloatTensor]):
    x = torch.einsum("...j,kj->...k", tensor, weight)
    if bias is not None:
        x += bias
    return x


def test_cases14():
    out = []
    for tensor in test_cases12():
        j = tensor.shape[-1]
        k = np.random.randint(10, 100)
        weight = torch.randn((k, j))
        bias = torch.randn((k,))
        out += [[tensor, weight, bias]]
    return out


def layer_norm(x: torch.FloatTensor, reduce_dims, weight: torch.FloatTensor, bias: torch.FloatTensor):
    red_dim_indices = list(range(len(x.shape) - len(reduce_dims), len(x.shape)))
    xmean = x.mean(dim=red_dim_indices, keepdim=True)
    var = ((x - xmean) ** 2).mean(dim=red_dim_indices, keepdim=True)
    xnorm = (x - xmean) / var.sqrt()
    return xnorm * weight + bias


def test_cases15():
    out = []
    for i in range(10):
        n_red_dims = np.random.randint(1, 3)
        n_batch_dims = np.random.randint(1, 3)
        reduce_dims = list(np.random.randint(10, 20, (n_red_dims,)))
        batch_dims = list(np.random.randint(20, 50, (n_batch_dims,)))
        x = torch.randn(batch_dims + reduce_dims)
        weight = torch.randn(reduce_dims)
        bias = torch.randn(reduce_dims)
        out += [[x, reduce_dims, weight, bias]]
    return out


def embed(x: torch.LongTensor, embeddings: torch.FloatTensor):
    return embeddings[x]


def test_cases16():
    out = []
    for i in range(10):
        vocab_size = np.random.randint(50, 100)
        embed_size = np.random.randint(8, 32)
        embeddings = torch.randn((vocab_size, embed_size))
        x_len = np.random.randint(10, 20)
        x = torch.randint(0, vocab_size, (x_len,))
        out += [[x, embeddings]]
    return out


def softmax(tensor: torch.FloatTensor):
    exps = torch.exp(tensor)
    return exps / exps.sum(dim=1, keepdim=True)


def test_cases17():
    out = []
    for i in range(10):
        n_inputs = np.random.randint(50, 100)
        n_classes = np.random.randint(3, 20)
        out += [torch.randn((n_inputs, n_classes))]
    return out


def logsoftmax(tensor: torch.FloatTensor):
    C = tensor.max(dim=1, keepdim=True).values
    return tensor - C - (tensor - C).exp().sum(dim=1, keepdim=True).log()


test_cases18 = test_cases17


def cross_entropy_loss(logits: torch.FloatTensor, y: torch.LongTensor):
    logprobs = logsoftmax(logits)
    return -torch.gather(logprobs, 1, y[:, None]).sum()


def test_cases19():
    out = []
    for logits in test_cases17():
        probs = softmax(logits)
        y = torch.randint(0, probs.shape[1], (probs.shape[0],))
        out += [[probs, y]]
    return out


def test_cases20():
    weight = torch.rand(5)
    data = torch.rand(22)
    return [[weight, data]]


def ex20(x, weight):
    kernel_size = weight.shape
    S = x.shape
    x = x.contiguous()
    strided_input = torch.as_strided(x, (S - kernel_size + 1, kernel_size), (S, 1), 0)
    added = strided_input * weight
    summed = reduce(added, "s k -> s", "sum")
    return summed


def test_cases21():
    weight = torch.rand(6, 5, 4)
    data = torch.rand(2, 22, 5)
    return [[weight, data]]


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


ex12 = relu
ex13 = dropout
ex14 = linear
ex15 = layer_norm
ex16 = embed
ex17 = softmax
ex18 = logsoftmax
ex19 = cross_entropy_loss
