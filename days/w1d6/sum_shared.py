import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from benchmark import benchmark
from pycuda_utils import Holder, load_module, ceil_divide

DEVICE = torch.device('cuda:0')

cuda_module = load_module("sum_shared.cu")
sum_shared_kernel = cuda_module.get_function("sum_shared")
sum_block_shfl_kernel = cuda_module.get_function("sum_block_shfl")

BLOCK_SIZE = 512
TYPE = torch.int32

def block_sum(vals):
    while True:
        size = vals.shape[0]
        if (size == 1):
            return vals.item()
        num_blocks = ceil_divide(size, BLOCK_SIZE)
        dest = torch.empty(num_blocks, dtype=TYPE, device=DEVICE)
        sum_shared_kernel(Holder(vals), np.int64(size), Holder(dest), 
               block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))
        vals = dest

def sum_block_shfl(vals):
    while True:
        size = vals.shape[0]
        if (size == 1):
            return vals.item()
        num_blocks = ceil_divide(size, BLOCK_SIZE)
        dest = torch.empty(num_blocks, dtype=TYPE, device=DEVICE)
        sum_block_shfl_kernel(Holder(vals), np.int64(size), Holder(dest), 
               block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))
        vals = dest


def run_benchmark(input_size):
    vals = torch.randint(1, 3, (input_size,), dtype=TYPE, device=DEVICE)
    time = benchmark(lambda: sum_block_shfl(vals), iters=3)
    print(f"size={input_size} time={time}")

    expected = vals.sum().item()
    result = block_sum(vals)
    # print("result", result)
    # print("expected", expected)
    assert result == expected

for size in [1000000, 100000000]:
    run_benchmark(size)
