import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module, ceil_divide
from benchmark_soln import benchmark

DEVICE = torch.device('cuda:0')

mod = load_module('sum_atomic_soln.cu')
sum_atomic_kernel = mod.get_function("sum_atomic")

def sum_atomic(inp: torch.Tensor, block_size: int = 512) -> torch.Tensor:
    inp = inp.to(DEVICE)

    dest = torch.zeros(1, device=DEVICE)

    sum_atomic_kernel(Holder(inp),
                      Holder(dest),
                      np.int32(inp.size(0)),
                      block=(block_size, 1, 1),
                      grid=(ceil_divide(inp.size(0), block_size), 1))
    torch.cuda.synchronize()

    return dest.cpu()


inp = torch.randn(100_000)
actual = sum_atomic(inp)
expected = inp.sum()
print(actual, expected, actual - expected)

for size in [10_000, 100_000, 1_000_000]:
    def fn():
        inp = torch.randn(size)
        sum_atomic(inp)
    t_per_iter = benchmark(fn)
    print(f'Size: {size}, elapsed: {t_per_iter:.2f}')

# TODO: plotting here