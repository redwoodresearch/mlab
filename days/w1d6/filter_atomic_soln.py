import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module, ceil_divide
from benchmark_soln import benchmark

DEVICE = torch.device('cuda:0')

mod = load_module('filter_atomic_soln.cu')
sum_atomic_kernel = mod.get_function("filter_atomic")

def filter_atomic(inp: torch.Tensor, dest: torch.Tensor, block_size: int = 512) -> torch.Tensor:
    counter = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    sum_atomic_kernel(Holder(inp),
                      Holder(dest),
                      Holder(counter),
                      np.int32(inp.size(0)),
                      np.float32(0.5),
                      block=(block_size, 1, 1),
                      grid=(ceil_divide(inp.size(0), block_size), 1))
    torch.cuda.synchronize()
    return dest, counter


inp = torch.randn(100).to(device=DEVICE)
dest = torch.zeros(100).to(device=DEVICE)
dest, counter = filter_atomic(inp, dest)
print(inp)
print(counter)
print(dest)
