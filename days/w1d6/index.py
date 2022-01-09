import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from torch._C import dtype

from pycuda_utils import Holder, load_module, ceil_divide

mod = load_module("index.cu")
index = mod.get_function("index")
size = np.uint32(123)
a = torch.randn(size, dtype=torch.float32).cuda()
b = torch.randperm(size, dtype=torch.int64).cuda()
dst = torch.zeros_like(a).cuda()

index(
    Holder(a),
    Holder(b),
    Holder(dst),
    size,
    block=(512, 1, 1),
    grid=(int(ceil_divide(size, 512)), 1),
)

# Kernels run async by default, so we call synchronize() before using values.
torch.cuda.synchronize()

print((dst == a[b]).float().mean())
# 0 2 6 12 20 30
