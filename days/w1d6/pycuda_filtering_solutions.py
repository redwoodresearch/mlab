import time
from typing import List, Tuple

import pycuda.autoinit as _
from pycuda.compiler import SourceModule
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from days.w1d6.pycuda_utils import Holder, ceil_divide

mod = SourceModule(open('days/w1d6/pycuda_filtering_solutions.cu').read(),
                   no_extern_c=True)

filter_atomic_kernel = mod.get_function("filter_atomic_kernel")

DEVICE = torch.device('cuda:0')


def filter_atomic(inp: torch.Tensor,
                  dest: torch.Tensor,
                  thresh: float,
                  block_size: int = 512) -> torch.Tensor:
    inp = inp.to(DEVICE)
    dest = dest.to(DEVICE)
    atomic = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    filter_atomic_kernel(Holder(inp),
                         Holder(dest),
                         Holder(atomic),
                         np.int32(inp.size(0)),
                         np.float32(thresh),
                         block=(block_size, 1, 1),
                         grid=(ceil_divide(inp.size(0), block_size), 1))
    torch.cuda.synchronize()

    return dest[:atomic]


def pred(x, thresh: float):
    return abs(x) % 1. < thresh


def filter_pytorch(
    inp: torch.Tensor,
    _: torch.Tensor,
    thresh: float,
) -> torch.Tensor:
    out = inp[pred(inp, thresh)]
    torch.cuda.synchronize()
    return out


def tensor_same_elems(l: torch.Tensor, r: torch.Tensor) -> bool:
    l_set = set(x.item() for x in l.flatten())
    r_set = set(x.item() for x in r.flatten())
    return l.size() == r.size() and l_set == r_set


def test_tensors(block_size: int = 512) -> List[torch.Tensor]:
    return [
        torch.tensor([1.7]),
        torch.tensor([3.7]),
        torch.tensor([-3.7]),
        torch.tensor([3., 2.4, -2., 3.9, 2.7, 2.1]),
        torch.rand(block_size - 1),
        torch.rand(block_size),
        torch.rand(block_size + 1),
        torch.rand(block_size + 7),
        torch.rand(round(1.5 * block_size)),
        torch.rand(2 * block_size - 1),
        torch.rand(10000),
    ]


def check_filterer(filterer, block_size: int = 512) -> None:
    for tensor in test_tensors(block_size=block_size):
        dest = tensor.clone()
        for thresh in [0.1, 0.3, 0.5, 0.9]:
            expected = filter_pytorch(tensor, dest, thresh)
            actual = filterer(tensor, dest, thresh)
            assert tensor_same_elems(expected, actual)


def benchmark_filterer(filterer,
                       size: int,
                       thresh: float,
                       iters: int = 10,
                       cpu=False) -> float:
    x = torch.rand(size)
    if not cpu:
        x = x.to(DEVICE)
    y = x.clone()

    for _ in range(3):
        filterer(x, y, thresh)

    start = time.time()
    for _ in range(iters):
        filterer(x, y, thresh)

    return (time.time() - start) / iters


def all_benchmarks(filterer,
                   max_size_power: int,
                   thresh: float,
                   cpu=False) -> Tuple[List[int], List[float]]:
    sizes = [2**x for x in range(6, max_size_power)]

    return sizes, [
        benchmark_filterer(filterer, s, thresh, 100 if s < 17 else 10, cpu=cpu)
        for s in sizes
    ]


if __name__ == "__main__":
    check_filterer(filter_atomic)

    end = 50

    def plot_benchs(filterer, max_size_power, thresh, label, cpu=False):
        sizes, times = all_benchmarks(filterer,
                                      min(max_size_power, end),
                                      thresh,
                                      cpu=cpu)
        sizes = np.array(sizes)
        times = np.array(times)
        plt.plot(sizes, sizes / times, label=label)

    threshs = [0.01, 0.05, 0.2, 0.5, 0.9]
    # threshs = [0.2]

    for thresh in tqdm(threshs):
        plot_benchs(filter_atomic, 26, thresh, f"atomic {thresh}")
        plot_benchs(filter_pytorch, 26, thresh, f"pytorch {thresh}")
        plot_benchs(filter_pytorch,
                    18,
                    thresh,
                    f"pytorch cpu {thresh}",
                    cpu=True)

    plt.ylabel("floats/sec")
    plt.legend()
    plt.show()
