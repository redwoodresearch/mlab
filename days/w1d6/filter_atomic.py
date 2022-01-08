import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module

cuda_module = load_module("filter_atomic.cu")
kernel = cuda_module.get_function("filter")
counter = torch.zeros(1, dtype=torch.int32).cuda()
input_values = torch.randn(2048, dtype=torch.float32).cuda()
input_size = input_values.shape[0]
output_values = torch.empty(2048, dtype=torch.float32).cuda()

BLOCK_SIZE = 256
num_blocks = (input_size + BLOCK_SIZE - 1) // 256
THRESHOLD = 0.1
kernel(Holder(input_values), np.int32(input_size), np.float32(THRESHOLD),
        Holder(output_values), Holder(counter),
        block=(BLOCK_SIZE, 1, 1), grid=(num_blocks, 1))

output_size = counter.item()
print("result size:", output_size)
print("result:", output_values[:output_size])
