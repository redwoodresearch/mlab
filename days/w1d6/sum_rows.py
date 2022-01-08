import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from benchmark import benchmark
from pycuda_utils import Holder, load_module

DEVICE = torch.device('cuda:0')

cuda_module = load_module("sum_rows.cu")
kernel = cuda_module.get_function("sum_rows")

def run_benchmark(num_cols, num_rows):
    input_values = torch.rand(num_cols, num_rows, dtype=torch.float32, device=DEVICE)
    BLOCK_SIZE = 256
    num_blocks = (num_cols + BLOCK_SIZE - 1) // 256;
    def run_kernel():
        dest = torch.empty(num_cols, dtype=torch.float32, device=DEVICE)
        kernel(Holder(input_values), np.int64(num_cols), np.int64(num_rows),
               Holder(dest),
               block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))
        torch.cuda.synchronize()
        return dest
    time = benchmark(run_kernel, iters=3)
    print("dims", num_cols, num_rows, ": time", time)

    expected = input_values.sum(dim=1)
    result = run_kernel()
    assert torch.allclose(result, expected)

for num_cols, num_rows in [(100000, 10), (10, 100000)]:
    run_benchmark(num_cols, num_rows)
