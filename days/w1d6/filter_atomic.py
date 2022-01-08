import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module


def ceil_div(a, b):
    return -(-a // b)


device = torch.device("cuda")


def filter_atomic(input, threshold):
    dest = torch.empty_like(input)
    ptr = torch.tensor(0, dtype=torch.int, device=device)
    BLOCKSIZE = 128
    gridsize = ceil_div(len(input), BLOCKSIZE)
    filter_atomic_kernel(
        Holder(dest),
        Holder(input),
        np.int32(len(input)),
        Holder(ptr),
        np.float32(threshold),
        block=(BLOCKSIZE, 1, 1),
        grid=(gridsize, 1, 1),
    )
    return dest[:ptr]


mod = load_module("filter_atomic.cu")
filter_atomic_kernel = mod.get_function("filter_atomic")


input = torch.rand(10, dtype=torch.float32, device=device)*2-1
threshold = 0.5
output = filter_atomic(input, threshold)

print(input)
print(output)
