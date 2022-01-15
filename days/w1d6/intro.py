import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module

mod = load_module('intro.cu')
zero_kernel = mod.get_function('zero')
one_kernel = mod.get_function('one')
dest = torch.ones(128, dtype=torch.float32).cuda()
zero_kernel(Holder(dest), block=(64, 1, 1), grid=(1, 1))

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()
print(dest)