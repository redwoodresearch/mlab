import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module, ceil_divide
from benchmark_soln import benchmark

DEVICE = torch.device('cuda:0')

mod = load_module('sum_rows_soln.cu')
kernel = mod.get_function("sum_rows")

block_size = 512
nCols = 200
nRows = 300

inp = torch.rand((nCols, nRows), device=DEVICE, dtype=torch.float32)
dest = torch.zeros(nCols, device=DEVICE, dtype=torch.float32)

kernel(Holder(dest),
            Holder(inp),
            np.int64(nCols),
            np.int64(nRows),
            block=(block_size, 1, 1),
            grid=(ceil_divide(dest.size(0), block_size), 1))
torch.cuda.synchronize()

print(dest)
expected = inp.sum(dim=-1)
print(expected)
assert torch.allclose(dest, expected)

