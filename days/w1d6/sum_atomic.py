import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from benchmark import benchmark
from pycuda_utils import Holder, load_module

cuda_module = load_module("sum_atomic.cu")
kernel = cuda_module.get_function("sum")

def run_benchmark(input_size):
    input_values = torch.randint(0, 3, (input_size,), dtype=torch.int32).cuda() * 3

    BLOCK_SIZE = 256
    num_blocks = (input_size + BLOCK_SIZE - 1) // 256;
    def run_kernel():
        dest = torch.zeros(1, dtype=torch.int32).cuda()
        kernel(Holder(input_values), np.int32(input_size), Holder(dest),
                block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))
        torch.cuda.synchronize()
        return dest.item()
    time = benchmark(run_kernel, iters=3)
    print(f"handwritten input_size={input_size} time={time}")

    def run_torch():
        return input_values.sum()
    time = benchmark(run_torch, iters=3)
    print(f"torch gpu input_size={input_size} time={time}")

    # expected_value = input_values.sum()
    # result = run_kernel()
    # if result != expected_value:
    #     err = f"expected={expected_value} actual={result}"
    #     raise AssertionError(err)

for size in [1000000, 100000000]:
    run_benchmark(size)
