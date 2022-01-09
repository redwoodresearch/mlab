import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from torch._C import dtype

from pycuda_utils import Holder, load_module, ceil_divide

mod = load_module("filter_atomic.cu")
filter = mod.get_function("filter")
src = torch.randn(10213, dtype=torch.float32).cuda()
size = np.uint32(src.size()[0])
dst = torch.zeros_like(src).cuda()
thresh = np.float32(0.5)
counter = torch.zeros(1, dtype=torch.int32).cuda()

filter(
    Holder(src),
    size,
    Holder(dst),
    thresh,
    Holder(counter),
    block=(512, 1, 1),
    grid=(int(ceil_divide(size, 512)), 1),
)

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()

print(counter, (src.abs() < thresh).sum())
print((dst.abs() >= thresh).sum())

print(src[src.abs() < thresh][:10])
print(dst[:10])
# 0 2 6 12 20 30
