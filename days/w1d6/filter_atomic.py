import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from math import ceil

from pycuda_utils import Holder, load_module, ceil_divide


mod = load_module('filter_atomic.cu')
filter_kernel = mod.get_function('filterAtomics')

SIZE = 100
dest = torch.zeros(SIZE, dtype=torch.float32).cuda()
values = torch.randn(SIZE, dtype=torch.float32).cuda()
counter = torch.zeros(1).int().cuda()
threshold = np.float32(0.1)
input_size = np.int32(SIZE)
num_blocks = int(ceil_divide(SIZE, 512))


filter_kernel(Holder(dest), Holder(values), threshold, Holder(counter), input_size, block=(512, 1, 1), grid=(num_blocks, 1))
torch_ans = values[torch.abs(values) < threshold]

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()
print(dest)
print(torch_ans)
print(torch.isclose(dest[:len(torch_ans)], torch_ans))