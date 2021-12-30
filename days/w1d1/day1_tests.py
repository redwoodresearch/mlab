import einops
import math
from einops.einops import rearrange, reduce
import numpy as np
from scipy.stats import chisquare
import torch
from typing import Collection, Optional


def _allclose_tensorlists(tensorlist1, tensorlist2):
    if isinstance(tensorlist1, torch.Tensor) and isinstance(tensorlist2, torch.Tensor):
        tensorlist1 = [tensorlist1]
        tensorlist2 = [tensorlist2]
    for t1, t2 in zip(tensorlist1, tensorlist2):
        if t1.shape != t2.shape:
            return False
        if not torch.allclose(t1, t2):
            return False
    return True


def _set_seeds(seed):
    rs = np.random.RandomState(seed)
    torch.manual_seed(seed)
    return rs


def _rand_shape(rs: np.random.RandomState, n_dims: Collection[int]):
    n_dim = rs.choice(n_dims)
    return list(rs.randint(3, 6, n_dim))


def _rand_tensor(shape: list, std=10.0):
    return (torch.randn(shape) * std).round()


def _count(sample, keys):
    counts = dict(zip(*np.unique(sample, return_counts=True)))
    return [counts.get(i, 0) for i in keys]


def test(f, ex_num, n_tests=10):
    g = globals()
    ex = g.get(f"ex{ex_num}")
    testcase = g.get(f"testcase{ex_num}")
    assert ex, f"There's no solution for exercise {ex_num}"

    deterministic_exercises = {
        1,
        2,
        3,
        5,
        7,
        8,
        9,
        11,
        12,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    }

    if ex_num in deterministic_exercises:
        for seed in range(n_tests):
            tc = testcase(seed)
            if not _allclose_tensorlists(
                f(*tc), ex(*tc)
            ):  # most of these funcs don't return lists??
                print(f"Wrong answer for testcase{ex_num}({seed}).")
                return

    if ex_num == 4:
        for seed in range(n_tests):
            tc = testcase(seed)
            ex_coords = {(xy[0].item(), xy[1].item()) for xy in ex(*tc)}
            f_coords = {(xy[0].item(), xy[1].item()) for xy in f(*tc)}
            if f_coords != ex_coords:
                print(f"Wrong answer for testcase{ex_num}({seed}).")
                return

    if ex_num == 6:
        for seed in range(n_tests):
            tc = testcase(seed)
            frequencies = _count(f(*tc), range(len(tc[1])))
            chisq, p = chisquare(frequencies, tc[0] * tc[1], axis=None)
            if p < 0.01:
                print(
                    "Your function returned an unexpected sample for "
                    f"testcase{ex_num}({seed}). (p-value = {p})"
                )
                return

    if ex_num == 10:
        for seed in range(n_tests):
            A, n = testcase(seed)
            B = f(A, n)
            for i, (Arow, Brow) in enumerate(zip(A, B)):
                expected = _count(Arow, {a.item() for a in Arow})
                observed = _count(Brow, {a.item() for a in Arow})
                chisq, p = chisquare(
                    observed, expected / sum(expected) * sum(observed), axis=None
                )
                if p < 0.01:
                    print(
                        "Your function returned an unexpected sample for "
                        f"testcase{ex_num}({seed}), row {i}. (p-value = {p})"
                    )
                    return

    if ex_num == 13:
        for seed in range(n_tests):
            tensor, dropout_p, is_train = testcase(seed)
            if is_train:
                out = f(tensor, dropout_p, is_train)
                is_zero = out == 0.0
                if not torch.allclose(
                    out[~is_zero], tensor[~is_zero] / (1 - dropout_p)
                ):
                    print(f"Wrong answer for testcase{ex_num}({seed}).")
                    return

                if abs(is_zero.sum() - dropout_p * len(tensor.flatten())) > 5:
                    print(
                        "Your function zeroed out an unexpected number of elements for "
                        f"testcase{ex_num}({seed})."
                    )
                    return
            else:
                if not torch.allclose(tensor, f(tensor, dropout_p, is_train)):
                    print(f"Wrong answer for testcase{ex_num}({seed}).")
                    return

    print(f"Your function passed {n_tests} tests for example {ex_num}. Congrats!")
    return


############################################################################


def ex1():
    return [
        einops.rearrange(torch.arange(3, 9), "(h w) -> h w", h=3, w=2),
        einops.rearrange(torch.arange(1, 7), "(h w) -> h w", h=2, w=3),
        einops.rearrange(torch.arange(1, 7), "a -> 1 a 1"),
    ]


def testcase1(seed=0):
    return []


def ex2(temp: torch.Tensor):
    assert len(temp) % 7 == 0, "Input length must be a multiple of 7."

    weekly = einops.rearrange(temp, "(h w) -> h w", w=7)
    weekly_means = einops.reduce(temp, "(h 7) -> h", "mean")
    weekly_zeromean = weekly - weekly_means[:, None]
    weekly_normalised = weekly_zeromean / weekly.std(dim=1, keepdim=True)
    return [weekly_means, weekly_zeromean, weekly_normalised]


def testcase2(seed=0):
    rs = _set_seeds(seed)
    n_weeks = rs.randint(2, 5)
    return [torch.randint(30, 100, (n_weeks * 7,)).float()]


def ex3(tensor1: torch.Tensor, tensor2: torch.Tensor):
    assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape."
    return (tensor1 * tensor2).sum(dim=-1)


def testcase3(seed=0):
    rs = _set_seeds(seed)
    shape = _rand_shape(rs, [2, 3])
    return [_rand_tensor(shape), _rand_tensor(shape)]


def ex4(H: float, W: float, n: int):
    xaxis = torch.linspace(0, H, n + 1)
    xtile = torch.tile(xaxis, dims=(n + 1, 1))
    yaxis = torch.linspace(0, W, n + 1)[:, None]
    ytile = torch.tile(yaxis, dims=(n + 1,))
    return torch.stack(
        [
            einops.rearrange(xtile, "h w -> (h w)"),
            einops.rearrange(ytile, "h w -> (h w)"),
        ]
    ).T


def testcase4(seed=0):
    rs = _set_seeds(seed)
    n = rs.randint(2, 10)
    w = rs.randint(5, 20)
    h = rs.randint(5, 20)
    return [h * n, w * n, n]


def ex5(n: int):
    matrix = (rearrange(torch.arange(n), "i->i 1") == torch.arange(n)).float()
    return matrix


def testcase5(seed=0):
    rs = _set_seeds(seed)
    return [rs.randint(2, 10)]


def ex6(n: int, probs: torch.Tensor):
    assert abs(probs.sum() - 1.0) < 0.001
    return (torch.rand((n, 1)) > torch.cumsum(probs, dim=0)).sum(dim=-1)


def test_fn_6(fn):
    probs = torch.nn.functional.softmax(t.rand(10), dim=-1)
    t = torch.stack([fn(probs) for _ in range(1000)], dim=0).sum(0)
    assert t.allclose(probs, t, atol=0.05, rtol=0.1)


def testcase6(seed=0):
    rs = _set_seeds(seed)
    n = 100
    k = rs.randint(3, 10)
    probs = rs.rand(k)
    probs /= probs.sum()
    probs = rs.multinomial(100, probs) / 100
    return [n, torch.Tensor(probs)]


def ex7(scores: torch.Tensor, y: torch.Tensor):
    return (scores.argmax(dim=1) == y).to(float).mean()


def testcase7(seed=0):
    rs = _set_seeds(seed)
    n_inputs = rs.randint(10, 20)
    n_classes = rs.randint(3, 8)
    scores = _rand_tensor((n_inputs, n_classes))
    y = torch.randint(n_classes, (n_inputs,))
    return [scores, y]


def ex8(scores: torch.Tensor, y: torch.Tensor, k: int):
    return (torch.argsort(scores)[:, -k:] == y[:, None]).any(dim=-1).to(float).mean()


def testcase8(seed=0):
    rs = _set_seeds(seed)
    n_inputs = rs.randint(10, 20)
    n_classes = rs.randint(4, 8)
    scores = _rand_tensor((n_inputs, n_classes))
    y = torch.randint(n_classes, (n_inputs,))
    k = rs.randint(2, n_classes // 2 + 1)
    return [scores, y, k]


def ex9(prices: torch.Tensor, items: torch.Tensor):
    return torch.gather(prices, 0, items.to(int)).sum()


def testcase9(seed=0):
    rs = _set_seeds(seed)
    n_items = rs.randint(5, 10)
    prices = (torch.rand(n_items) * 100).round()
    n_buys = rs.randint(30, 50)
    items = torch.randint(n_items, (n_buys,))
    return [prices, items]


def ex10(A: torch.Tensor, N: int):
    index = torch.randint(A.shape[-1], (A.shape[0], N))
    return torch.gather(A, 1, index)


def testcase10(seed=0):
    rs = _set_seeds(seed)
    m, k, n = rs.randint(5, 10, (3,))
    A = _rand_tensor((m, k))
    return [A, n]


def ex11(T: torch.Tensor, K: int, values: Optional[torch.Tensor] = None):
    if values is None:
        values = torch.ones(T.shape[0])
    onehot = torch.zeros(T.shape + (K,))
    return onehot.scatter(-1, T.to(int).unsqueeze(-1), values.unsqueeze(-1))


def testcase11(seed=0):
    rs = _set_seeds(seed)
    K = rs.randint(5, 10)
    shape = _rand_shape(rs, [1, 2])
    values = _rand_tensor(shape, 100.0)
    return [torch.randint(K, shape), K, values]


def relu(tensor: torch.FloatTensor) -> torch.Tensor:
    tensor = tensor.clone()
    tensor[tensor < 0] = 0
    return tensor


def testcase12(seed=0):
    rs = _set_seeds(seed)
    shape = _rand_shape(rs, [1, 2, 3])
    return [_rand_tensor(shape)]


def dropout(tensor: torch.FloatTensor, drop_fraction: float, is_train: bool):
    if is_train:
        mask = torch.rand_like(tensor) > drop_fraction
        return mask * tensor / (1 - drop_fraction)
    return tensor


def testcase13(seed=0):
    rs = _set_seeds(seed)
    shape = _rand_shape(rs, [1, 2, 3])
    drop_fraction = np.round(rs.rand() * 0.6 + 0.2, 2)
    is_train = bool(rs.randint(0, 2))
    return [_rand_tensor(shape), drop_fraction, is_train]


def linear(
    tensor: torch.FloatTensor,
    weight: torch.FloatTensor,
    bias: Optional[torch.FloatTensor],
):
    x = torch.einsum("...j,kj->...k", tensor, weight)
    if bias is not None:
        x += bias
    return x


def testcase14(seed=0):
    rs = _set_seeds(seed)
    shape = _rand_shape(rs, [2, 3])
    j = shape[-1]
    k = rs.randint(3, 10)
    weight = _rand_tensor((k, j))
    bias = _rand_tensor((k,))
    tensor = _rand_tensor(shape)
    return [tensor, weight, bias]


def layer_norm(
    x: torch.FloatTensor,
    reduce_dims,
    weight: torch.FloatTensor,
    bias: torch.FloatTensor,
):
    red_dim_indices = list(range(len(x.shape) - len(reduce_dims), len(x.shape)))
    xmean = x.mean(dim=red_dim_indices, keepdim=True)
    var = ((x - xmean) ** 2).mean(dim=red_dim_indices, keepdim=True)
    xnorm = (x - xmean) / var.sqrt()
    return xnorm * weight + bias


def testcase15(seed=0):
    rs = _set_seeds(seed)
    reduce_dims = _rand_shape(rs, [1, 2])
    batch_dims = _rand_shape(rs, [1, 2])
    x = _rand_tensor(batch_dims + reduce_dims)
    weight = _rand_tensor(reduce_dims)
    bias = _rand_tensor(reduce_dims)
    return [x, reduce_dims, weight, bias]


def embed(x: torch.LongTensor, embeddings: torch.FloatTensor):
    return embeddings[x]


def testcase16(seed=0):
    rs = _set_seeds(seed)
    vocab_size = rs.randint(8, 16)
    embed_size = rs.randint(4, 8)
    embeddings = _rand_tensor((vocab_size, embed_size))
    x_len = rs.randint(10, 20)
    x = torch.randint(0, vocab_size, (x_len,))
    return [x, embeddings]


def softmax(tensor: torch.FloatTensor):
    exps = torch.exp(tensor)
    return exps / exps.sum(dim=-1, keepdim=True)


def testcase17(seed=0):
    rs = _set_seeds(seed)
    n_inputs = rs.randint(10, 20)
    n_classes = rs.randint(3, 8)
    return [_rand_tensor((n_inputs, n_classes))]


def logsoftmax(tensor: torch.FloatTensor):
    C = tensor.max(dim=1, keepdim=True).values
    # subtracting max from both sides to avoid exponential going above the floating point range
    return tensor - C - (tensor - C).exp().sum(dim=-1, keepdim=True).log()


testcase18 = testcase17


def cross_entropy_loss(logits: torch.FloatTensor, y: torch.LongTensor):
    logprobs = logsoftmax(logits)
    return -torch.gather(logprobs, 1, y[:, None]).mean()


def testcase19(seed=0):
    rs = _set_seeds(seed)
    logits = testcase17(seed)[0]
    y = torch.randint(0, logits.shape[1], (logits.shape[0],))
    return [logits, y]


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


def testcase20(seed=0):
    rs = _set_seeds(seed)
    weight = _rand_tensor(5)
    data = _rand_tensor(22)
    return [data, weight]


def ex21(x, weight):
    x = rearrange(x, "b c s -> b s c")
    out_channels, _, kernel_size = weight.shape
    B, S, C = x.shape
    x = x.contiguous()
    strided_input = torch.as_strided(
        x,
        (B, S - kernel_size + 1, kernel_size, C, out_channels),
        (S * C, C, C, 1, 0),
        0,
    )
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


def testcase21(seed=0):
    rs = _set_seeds(seed)
    n_kernels = rs.randint(4, 8)
    n_channels = 5
    kernel_len = 4
    weight = _rand_tensor((n_kernels, n_channels, kernel_len))
    batch_size = rs.randint(8, 16)
    data_len = 22
    data = _rand_tensor((batch_size, n_channels, data_len))
    return [data, weight]


def ex22(v, w, padding=0, stride=1):
    batch_size, in_chans, seq_len = v.shape
    kernel_size = w.shape[-1]
    outlen = int((seq_len - kernel_size + padding * 2) / stride + 1)
    v = torch.cat(
        [
            torch.zeros(batch_size, in_chans, padding),
            v,
            torch.zeros(batch_size, in_chans, padding),
        ],
        dim=-1,
    )
    u = v.as_strided(
        (batch_size, outlen, in_chans, kernel_size),
        (in_chans * seq_len, stride, seq_len, 1),
    )
    return torch.einsum("btik, oik -> bot", [u, w])


def testcase22(seed=0):
    rs = _set_seeds(seed)
    n_kernels = rs.randint(4, 8)
    n_channels = 5
    kernel_len = 3
    weight = _rand_tensor((n_kernels, n_channels, kernel_len))
    batch_size = rs.randint(8, 16)
    data_len = 22
    data = _rand_tensor((batch_size, n_channels, data_len))
    padding = (kernel_len - 1) / 2
    stride = 2
    return [data, weight, padding, stride]


ex12 = relu
ex13 = dropout
ex14 = linear
ex15 = layer_norm
ex16 = embed
ex17 = softmax
ex18 = logsoftmax
ex19 = cross_entropy_loss
