import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module

cuda_module = load_module("abc.cu")
kernel = cuda_module.get_function("dot")
dest = torch.ones(128, dtype=torch.float32).cuda()
a = torch.rand(128, dtype=torch.float32).cuda()
b = torch.rand(128, dtype=torch.float32).cuda()
c = torch.rand(128, dtype=torch.float32).cuda()
expected_result = a * b + c

result = torch.zeros(128, dtype=torch.float32).cuda()
kernel(Holder(result), Holder(a), Holder(b), Holder(c), block=(128, 1, 1), grid=(1, 1))

torch.cuda.synchronize()
print("######### result")
print(result)
print("######### expected")
print(expected_result)
