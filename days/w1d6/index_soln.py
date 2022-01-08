import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module, ceil_divide
from benchmark_soln import benchmark

DEVICE = torch.device('cuda:0')

mod = load_module('index_soln.cu')
index_kernel = mod.get_function("index")

block_size = 512
aSize = 10
bSize = 100
a = torch.randn(aSize).to(device=DEVICE)
b = torch.randint(-aSize, aSize, size=(bSize,), dtype=torch.int64, device=DEVICE)
dest = torch.zeros(bSize).to(device=DEVICE)


index_kernel(Holder(dest),
            Holder(a),
            Holder(b),
            np.int64(aSize),
            np.int64(bSize),
            block=(block_size, 1, 1),
            grid=(ceil_divide(dest.size(0), block_size), 1))
torch.cuda.synchronize()

print(dest)

expected = a[b] 
expected[(b < 0) | (b >= aSize)] = 0
print(expected)

