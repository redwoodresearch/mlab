import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from math import ceil

from pycuda_utils import Holder, load_module, ceil_divide


mod = load_module('index.cu')
index_kernel = mod.get_function('index')

A_SIZE = 100
B_SIZE = 200
dest = torch.zeros(B_SIZE, dtype=torch.float32).cuda()
a = torch.randn(A_SIZE, dtype=torch.float32).cuda()
b = torch.floor(A_SIZE * torch.rand(B_SIZE)).long().cuda()
a_size = np.int32(A_SIZE)
b_size = np.int32(B_SIZE)
num_blocks = int(ceil_divide(B_SIZE, 512))

index_kernel(Holder(dest), Holder(a), a_size, Holder(b), b_size, block=(512, 1, 1), grid=(num_blocks, 1))
torch_ans = a[b]

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()
print(dest)
print(torch_ans)
print(torch.isclose(dest, torch_ans))