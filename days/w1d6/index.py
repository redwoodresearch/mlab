import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module, ceil_divide

DEVICE = torch.device('cuda:0')

cuda_module = load_module("index.cu")
kernel = cuda_module.get_function("index")
a = torch.arange(0, 12800, 100, dtype=torch.int32).cuda()
a_size = a.shape[0]
b = torch.randint(0, 128, (2048,), dtype=torch.int64).cuda()
# b = torch.randint(0, 150, (2048,), dtype=torch.int64).cuda() # expect out of bounds
b_size = b.shape[0]
output = torch.empty(b_size, dtype=torch.int32).cuda()
expected_output = None
if torch.all(b < a_size):
    expected_output = a[b]

BLOCK_SIZE = 256
num_blocks = ceil_divide(b_size, BLOCK_SIZE)
kernel(Holder(a), np.int64(a_size), Holder(b), np.int64(b_size), Holder(output),
        block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))

print("result:")
print(output)
print("expected:")
print(expected_output)
