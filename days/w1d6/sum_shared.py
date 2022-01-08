import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from benchmark import benchmark
from pycuda_utils import Holder, load_module, ceil_divide

DEVICE = torch.device('cuda:0')

cuda_module = load_module("sum_shared.cu")
kernel = cuda_module.get_function("sum_shared")

def run_benchmark(input_size):
    BLOCK_SIZE = 512
    assert input_size % BLOCK_SIZE == 0
    input_values = torch.randint(1, 3, (input_size,), dtype=torch.int32, device=DEVICE)
    num_blocks = ceil_divide(input_size, BLOCK_SIZE)
    def run_kernel():
        dest = torch.empty(num_blocks, dtype=torch.int32, device=DEVICE)
        kernel(Holder(input_values), Holder(dest), block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))
        torch.cuda.synchronize()
        return dest
    time = benchmark(run_kernel, iters=1)
    print("time", time)

    expected = input_values.sum()
    result = run_kernel().sum()
    print("result", result)
    print("expected", expected)
    assert result == expected

run_benchmark(5120)
