import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module

mul_add_mod = load_module('abc_soln.cu')
mul_add_kernel = mul_add_mod.get_function("mul_add")

size = 128
dest = torch.empty(size, dtype=torch.float32).cuda()
a = torch.arange(size).to(torch.float32).cuda()
b = torch.arange(0, 4 * size, 4).to(torch.float32).cuda()
c = torch.arange(3, size + 3).to(torch.float32).cuda()
mul_add_kernel(Holder(dest),
               Holder(a),
               Holder(b),
               Holder(c),
               block=(size, 1, 1),
               grid=(1, 1))
torch.cuda.synchronize()
print(dest)

assert torch.allclose(dest, a * b + c)