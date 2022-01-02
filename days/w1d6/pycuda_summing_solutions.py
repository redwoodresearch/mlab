import time
from typing import List, Tuple

import pycuda.autoinit as _
from pycuda.compiler import SourceModule
import torch
import numpy as np
import matplotlib.pyplot as plt

from days.w1d6.pycuda_utils import Holder, ceil_divide

mod = SourceModule(open('days/w1d6/pycuda_summing_solutions.cu').read(),
                   no_extern_c=True)

sum_atomic_kernel = mod.get_function("sum_atomic_kernel")
simple_sum_block_kernel_512 = mod.get_function("simple_sum_block_kernel_512")
shfl_reduce_kernel = mod.get_function("shfl_reduce_kernel")

DEVICE = torch.device('cuda:0')


def check_inp(inp: torch.Tensor) -> None:
    assert inp.dim() == 1
    assert inp.dtype == torch.float32


def sum_atomic(inp: torch.Tensor, block_size: int = 512) -> torch.Tensor:
    check_inp(inp)

    inp = inp.to(DEVICE)

    dest = torch.zeros(1, device=DEVICE)

    sum_atomic_kernel(Holder(inp),
                      Holder(dest),
                      np.int32(inp.size(0)),
                      block=(block_size, 1, 1),
                      grid=(ceil_divide(inp.size(0), block_size), 1))
    torch.cuda.synchronize()

    return dest.cpu()


def simple_sum_per_block(inp: torch.Tensor) -> torch.Tensor:
    check_inp(inp)

    block_size = 512
    inp = inp.to(DEVICE)
    n_blocks = ceil_divide(inp.size(0), block_size)
    dest = torch.empty(n_blocks, dtype=torch.float32, device=DEVICE)
    simple_sum_block_kernel_512(Holder(inp),
                                Holder(dest),
                                np.int32(inp.size(0)),
                                block=(block_size, 1, 1),
                                grid=(n_blocks, 1))
    torch.cuda.synchronize()

    return dest.cpu()


def simple_sum_segments(inp: torch.Tensor) -> torch.Tensor:
    check_inp(inp)

    block_size = 512
    inp = inp.to(DEVICE)
    n_blocks = ceil_divide(inp.size(0), block_size)
    dest_l = torch.empty(n_blocks, dtype=torch.float32, device=DEVICE)
    simple_sum_block_kernel_512(Holder(inp),
                                Holder(dest_l),
                                np.int32(inp.size(0)),
                                block=(block_size, 1, 1),
                                grid=(n_blocks, 1))
    sub_size = n_blocks
    dest_r = torch.empty(ceil_divide(n_blocks, block_size),
                         dtype=torch.float32,
                         device=DEVICE)
    torch.cuda.synchronize()

    curr_in, curr_out = dest_l, dest_r

    while sub_size > 1:
        next_sub_size = ceil_divide(sub_size, block_size)
        simple_sum_block_kernel_512(Holder(curr_in),
                                    Holder(curr_out),
                                    np.int32(sub_size),
                                    block=(block_size, 1, 1),
                                    grid=(next_sub_size, 1))
        torch.cuda.synchronize()
        curr_in, curr_out = curr_out, curr_in
        sub_size = next_sub_size

    return curr_in[0].cpu()


def shfl_reduce(inp: torch.Tensor,
                block_size: int = 512,
                max_grid: int = 1024) -> torch.Tensor:
    check_inp(inp)

    inp = inp.to(DEVICE)
    n_blocks = min(ceil_divide(inp.size(0), block_size), max_grid)

    dest = torch.empty(n_blocks, dtype=torch.float32, device=DEVICE)

    shfl_reduce_kernel(Holder(inp),
                       Holder(dest),
                       np.int32(inp.size(0)),
                       block=(block_size, 1, 1),
                       grid=(n_blocks, 1))
    torch.cuda.synchronize()
    shfl_reduce_kernel(Holder(dest),
                       Holder(dest),
                       np.int32(n_blocks),
                       block=(max_grid, 1, 1),
                       grid=(1, 1))
    torch.cuda.synchronize()

    return dest[0].cpu()


def pytorch_reduce(inp: torch.Tensor) -> torch.Tensor:
    return inp.sum().cpu()


def test_tensors(block_size: int = 512) -> List[torch.Tensor]:
    return [
        torch.tensor([1.7]),
        torch.tensor([1.2, 0., 123.]),
        torch.rand(block_size - 1),
        torch.rand(block_size),
        torch.rand(block_size + 1),
        torch.rand(block_size + 7),
        torch.rand(round(1.5 * block_size)),
        torch.rand(2 * block_size - 1),
        torch.rand(100000),
    ]


def check_reducer(reducer, block_size: int = 512) -> None:
    for tensor in test_tensors(block_size=block_size):
        assert torch.isclose(tensor.sum(), reducer(tensor))


def benchmark_reducer(reducer, size: int, iters: int = 10, cpu=False) -> float:
    x = torch.rand(size)
    if not cpu:
        x = x.to(DEVICE)

    for _ in range(3):
        reducer(x)

    start = time.time()
    for _ in range(iters):
        reducer(x)

    return (time.time() - start) / iters


def all_benchmarks(reducer,
                   max_size_power: int,
                   cpu=False) -> Tuple[List[int], List[float]]:
    sizes = [2**x for x in range(6, max_size_power)]

    return sizes, [
        benchmark_reducer(reducer, s, 100 if s < 17 else 10, cpu=cpu)
        for s in sizes
    ]


def torch_block_reduce(inp: torch.Tensor, block_size: int) -> torch.Tensor:
    check_inp(inp)

    padding_size = ceil_divide(inp.size(0),
                               block_size) * block_size - inp.size(0)
    padded = torch.cat((inp, torch.zeros(padding_size)))

    return padded.reshape(-1, block_size).sum(dim=-1)


def check_block_reducer(block_reducer, block_size=512) -> None:
    for tensor in test_tensors(block_size=block_size):
        assert torch.isclose(torch_block_reduce(tensor, block_size),
                             block_reducer(tensor)).all()


if __name__ == "__main__":
    check_reducer(sum_atomic)

    check_block_reducer(simple_sum_per_block)

    check_reducer(simple_sum_segments)

    check_reducer(shfl_reduce)

    end = 50

    def plot_benchs(reducer, max_size_power, label, cpu=False):
        sizes, times = all_benchmarks(reducer,
                                      min(max_size_power, end),
                                      cpu=cpu)
        sizes = np.array(sizes)
        times = np.array(times)
        plt.plot(sizes, sizes / times, label=label)

    plot_benchs(sum_atomic, 22, "atomic")
    plot_benchs(simple_sum_segments, 28, "simple")
    plot_benchs(shfl_reduce, 30, "shfl")
    plot_benchs(pytorch_reduce, 24, "pytorch cpu", cpu=True)
    plot_benchs(pytorch_reduce, 30, "pytorch")

    plt.ylabel("floats/sec")
    plt.legend()
    plt.show()
