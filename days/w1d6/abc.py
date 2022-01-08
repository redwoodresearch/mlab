import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module


mod = load_module('abc.cu')
abc_kernel = mod.get_function('abc')

dest = torch.ones(128, dtype=torch.float32).cuda()
a = torch.randn(128, dtype=torch.float32).cuda()
b = torch.randn(128, dtype=torch.float32).cuda()
c = torch.randn(128, dtype=torch.float32).cuda()
abc_kernel(Holder(dest), Holder(a), Holder(b), Holder(c), block=(128, 1, 1), grid=(1, 1))
torch_ans = a * b + c

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()
print(dest)
print(torch_ans)
print(torch.isclose(dest, torch_ans))